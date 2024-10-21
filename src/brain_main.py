#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
import statistics
import csv
from tqdm import tqdm

from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, ByzantineLocalUpdate, test_inference
from utils import get_dataset, compose_weight, exp_details

from cache import ItemCache
from moving_average import MovingAverage

from airbench.model import make_net
from airbench.hyperparameters import hyp


if __name__ == '__main__':
    start_time = time.time()
    traning_times = []

    # define paths
    path_project = os.path.abspath('.')
    logger = SummaryWriter('./logs')

    args = args_parser()
    # exp_details(args)

    num_byzantines = (args.byzantines if args.byzantines <
                      args.score_byzantines else args.score_byzantines)

    # load dataset and user groups
    os.makedirs('./save', exist_ok=True)
    os.makedirs('./save/objects', exist_ok=True)
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if (args.model == 'cnn') and (args.dataset == 'cifar'):
        pass
    else:
        exit('Error: unrecognized model')
    # Make Model
    widths = hyp['net']['widths']
    batchnorm_momentum = hyp['net']['batchnorm_momentum']
    scaling_factor = hyp['net']['scaling_factor']
    global_model = make_net(widths, batchnorm_momentum, scaling_factor)
    global_model.train()
    # print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    test_loss_collect, test_acc_collect = [], []

    queue_len, updates_verified, updates_rejected, updates_rejected_ref = [], [], [], []

    commit_ignored = 0

    # Cache
    cache = ItemCache(min_counter=0, max_counter=args.stale, brain=True)

    # Moving Average
    wma = MovingAverage(args.window)

    # Update History
    update_history = {}

    # Byzantine Count
    malicious = [{"count": 0, "streak" : 0} for _ in range(args.num_users)]

    for epoch in tqdm(range(args.epochs + args.stale)):
        if (len(cache.cache) == 0) and (epoch >= args.epochs):
            break
        
        updates_verified.append(0)
        updates_rejected.append(0)
        updates_rejected_ref.append(0)
        
        # local_weights = []
        # print(f'\n | Global Training Round : {epoch+1} |\n')

        if (epoch < args.epochs and len(cache.queue) <= args.maxqueue):
            global_model.train()
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(
                range(args.num_users), m, replace=False)

            optim_model = None
            ref = []

            if args.optim:
                optim_model = copy.deepcopy(global_model)
                
                count = len(cache.queue)

                if (count != 0):
                    optim_weight = copy.deepcopy(global_weights)
                    for id, item, _ in cache.queue:
                        proposer = update_history[id]

                        if streak:= malicious[proposer]["streak"] > 0:
                            commit_ignored += 1
                            continue

                        ref.append(id)

                        alpha = wma.current()
                        optim_weight = compose_weight(optim_weight, item, alpha)

                    optim_model.load_state_dict(optim_weight)

            for idx in idxs_users:
                if idx >= args.byzantines:
                    local_model = LocalUpdate(args=args, hyps=hyp,
                                              dataset=train_dataset, idxs=user_groups[idx -
                                                                                      num_byzantines],
                                              logger=logger)
                    traning_start = time.time()
                else:
                    local_model = ByzantineLocalUpdate(args=args, hyps=None,
                                                       dataset=train_dataset, idxs=[],
                                                       logger=logger)

                w, loss = local_model.update_weights(
                    model=copy.deepcopy(optim_model if args.optim else global_model), epochs=args.local_ep, global_round=epoch)
                if idx >= args.byzantines:
                    traning_times.append(time.time() - traning_start)

                # local_weights.append(copy.deepcopy(w))
                id = cache.add_item_with_random_counter(copy.deepcopy(w), ref)

                update_history[id] = idx

        target_id, local_weight = cache.update_counters()

        # BRAIN: do evaluate, to get score, among randomly sampled nodes
        committee = list(range(args.num_users))
        if args.diff != 1.0:
            m = max(int(args.diff * args.num_users), 1)
            committee = np.random.choice(
                range(args.num_users), m, replace=False)

        local_eval_med_acc = 0
        if target_id:
                local_eval_acc = []

                for idx in committee:
                    # BRAIN: `score_byzantines` submit random score
                    if idx >= args.score_byzantines:
                        local_model = LocalUpdate(args=args, hyps=hyp,
                                                  dataset=train_dataset, idxs=user_groups[idx -
                                                                                          num_byzantines],
                                                  logger=logger)
                        temp_model = copy.deepcopy(global_model)
                        temp_model.load_state_dict(local_weight)
                        temp_model.eval()
                        acc, loss = local_model.inference(model=temp_model)
                        local_eval_acc.append(acc)
                    else:
                        local_eval_acc.append(np.random.random())

                med_score = statistics.median(local_eval_acc)
                # BRAIN: reject updates using score by `threshold`
                if med_score >= args.threshold:
                    local_eval_med_acc = med_score
                else:
                    local_eval_med_acc = None

        # update global weights
        if target_id:

            proposer = update_history[target_id]

            if local_eval_med_acc is None:
                updates_rejected[-1] += 1
                
                if args.optim:
                    updates_rejected_ref[-1] += len(cache.queue) + len(cache.cache)

                    cache.cache = [(id_, item_, counter, ref_) for id_, item_, counter, ref_ in cache.cache if target_id not in ref_]
                    cache.queue = [(id_, item_, ref_) for id_, item_, ref_ in cache.queue if target_id not in ref_]
                    
                    updates_rejected_ref[-1] -= len(cache.queue) + len(cache.cache)

                    updates_rejected[-1] += updates_rejected_ref[-1]

                if args.history:
                    malicious[proposer]["count"] = malicious[proposer]["count"] + 1
                    malicious[proposer]["streak"] = malicious[proposer]["count"]
            else:
                updates_verified[-1] += 1

                if args.history:
                    if streak:= malicious[proposer]["streak"] > 0:
                            malicious[proposer]["streak"] = streak - 1
                
                alpha = wma.next(local_eval_med_acc)
                global_weights = compose_weight(
                    global_weights, local_weight, alpha)
                global_model.load_state_dict(global_weights)

        # Test inference after completion of training
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        test_acc_collect.append(test_acc)
        test_loss_collect.append(test_loss)

        queue_len.append(len(cache.queue))

        # print(
        #     f'\nResults after {epoch+1}/{args.epochs+1} global rounds of training:')
        # print("Test Accuracy: {:.2f}%".format(100*test_acc))
        # print(f'Test Loss    : {format(test_loss)}')

    argstring = 'brain{}_{}_{}_{}_C{}_iid{}_E{}_B{}_Z{}_SZ{}_D{}_W{}_S{}_TH{}_Q{}{}'.\
        format('_optim' if args.optim else '', args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs, args.byzantines, args.score_byzantines,
               args.diff, args.window, args.stale, args.threshold, args.maxqueue, '_H' if args.history else '')
    
    header = argstring + '_' + str(time.time())

    # Saving the objects test_loss_collect and test_acc_collect:
    with open('./save/objects/' + header + '.pkl', 'wb') as f:
        pickle.dump([test_loss_collect, test_acc_collect], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
    print(f'\n Avg Training Time: {np.median(np.array(traning_times))}')
    # file_path = './results/times.csv'
    # os.makedirs('./results', exist_ok=True)
    # with open(file_path, 'a', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(traning_times)

    print(" Average Acc: {}".format(np.mean(test_acc_collect)))
    print(" Peak Acc: {}\n".format(np.max(test_acc_collect)))

    if args.optim and args.history:
        print(" {} Commits ignored during lookahead model update\n".format(commit_ignored))

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    png_path_format = './save/' + argstring + '/{}/'
    png_file_format = png_path_format + header + '_{}.png'

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(test_loss_collect)), test_loss_collect, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    os.makedirs(png_path_format.format('loss'), exist_ok=True)
    plt.savefig(png_file_format.format('loss', 'loss'))

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(test_acc_collect)), test_acc_collect, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    os.makedirs(png_path_format.format('acc'), exist_ok=True)
    plt.savefig(png_file_format.format('acc', 'acc'))

    # Plot Queue Length vs Communication rounds
    plt.figure()
    plt.title('Queue Length vs Communication rounds')
    plt.plot(range(len(queue_len)), queue_len, color='k')
    plt.ylabel('Queue Length')
    plt.xlabel('Communication Rounds')
    os.makedirs(png_path_format.format('queue'), exist_ok=True)
    plt.savefig(png_file_format.format('queue', 'queue'))

    # Plot Cumulative Commit Results vs Communication rounds
    plt.figure()
    plt.title('Commit Results vs Communication rounds')
    plt.stackplot(range(len(updates_verified)), np.cumsum(updates_verified), np.cumsum(updates_rejected), colors=['b', 'r'])
    plt.ylabel('Commit Results')
    plt.xlabel('Communication Rounds')
    os.makedirs(png_path_format.format('commit'), exist_ok=True)
    plt.savefig(png_file_format.format('commit', 'commit'))
