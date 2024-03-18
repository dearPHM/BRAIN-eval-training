#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
import statistics
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, ByzantineLocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, compose_weight, exp_details

from cache import ItemCache
from moving_average import MovingAverage


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('.')
    logger = SummaryWriter('./logs')

    args = args_parser()
    exp_details(args)

    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    # Cache
    cache = ItemCache(min_counter=0, max_counter=args.stale)

    # Moving Average
    wma = MovingAverage(args.window)

    for epoch in tqdm(range(args.epochs + args.stale)):
        if (len(cache.cache) == 0) and (epoch >= args.epochs):
            break

        local_weights = []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        if (epoch < args.epochs):
            global_model.train()
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(
                range(args.num_users), m, replace=False)

            for idx in idxs_users:
                if idx >= args.byzantines:
                    local_model = LocalUpdate(args=args, dataset=train_dataset,
                                              idxs=user_groups[idx], logger=logger)
                else:
                    local_model = ByzantineLocalUpdate(args=args, dataset=train_dataset,
                                                       idxs=user_groups[idx], logger=logger)
                w, loss = local_model.update_weights(
                    model=copy.deepcopy(global_model), global_round=epoch)
                # local_weights.append(copy.deepcopy(w))
                cache.add_item_with_random_counter(copy.deepcopy(w))

        local_weights = cache.update_counters()

        # BRAIN: do evaluate, to get score, among randomly sampled nodes
        m = max(int(args.diff * args.num_users), 1)
        committee = np.random.choice(
            range(args.num_users), m, replace=False)

        local_eval_med_accs = []
        if len(local_weights) != 0:
            for local_weight in local_weights:
                local_eval_acc = []

                for idx in committee:
                    # BRAIN: `score_byzantines` submit random score
                    if idx >= args.score_byzantines:
                        local_model = LocalUpdate(args=args, dataset=train_dataset,
                                                  idxs=user_groups[idx], logger=logger)
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
                    local_eval_med_accs.append(med_score)
                else:
                    local_eval_med_accs.append(None)

        # update global weights
        if len(local_weights) != 0:
            for local_weight, score in zip(local_weights, local_eval_med_accs):
                # BRAIN: aggregate updates using `window`-sized moving average
                if score is None:
                    pass
                else:
                    alpha = wma.next(score)
                    global_weights = compose_weight(
                        global_weights, local_weight, alpha)
                    global_model.load_state_dict(global_weights)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            # if idx >= args.byzantines:
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))
        train_loss.append(sum(list_loss)/len(list_loss))

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = './save/objects/brain_{}_{}_{}_C{}_iid{}_E{}_B{}_Z{}_SZ{}_S{}_TH{}_{}.pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs, args.byzantines, args.score_byzantines, args.stale, args.threshold, time.time())

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/brain_{}_{}_{}_C{}_iid{}_E{}_B{}_Z{}_SZ{}_S{}_TH{}_loss.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs, args.byzantines, args.score_byzantines, args.stale, args.threshold))

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/brain_{}_{}_{}_C{}_iid{}_E{}_B{}_Z{}_SZ{}_S{}_TH{}_acc.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs, args.byzantines, args.score_byzantines, args.stale, args.threshold))
