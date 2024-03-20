#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, ByzantineLocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, compose_weight, exp_details

from cache import ItemCache


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('.')
    logger = SummaryWriter('./logs')

    args = args_parser()
    exp_details(args)

    device = torch.device('cuda' if (
        args.gpu != None and torch.cuda.is_available()) else 'cpu')
    if device == 'cuda':
        torch.cuda.set_device(args.gpu)

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
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    test_loss_collect, test_acc_collect = [], []

    # Cache
    cache = ItemCache(min_counter=0, max_counter=args.stale)

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

        # update global weights
        if len(local_weights) != 0:
            for local_weight in local_weights:
                global_weights = compose_weight(
                    global_weights, local_weight, args.alpha)
                global_model.load_state_dict(global_weights)

        # # Calculate avg training accuracy over all users at every epoch
        # list_acc, list_loss = [], []
        # global_model.eval()
        # for idx in idxs_users:
        #     local_model = LocalUpdate(args=args, dataset=train_dataset,
        #                               idxs=user_groups[idx], logger=logger)
        #     acc, loss = local_model.inference(model=global_model)
        #     # if idx >= args.byzantines:
        #     list_acc.append(acc)
        #     list_loss.append(loss)
        # test_acc_collect.append(sum(list_acc)/len(list_acc))
        # test_loss_collect.append(sum(list_loss)/len(list_loss))

        # # print global training loss after every 'i' rounds
        # if (epoch+1) % print_every == 0:
        #     print(f' \nAvg Training Stats after {epoch+1} global rounds:')
        #     print(f'Training Loss : {np.mean(np.array(test_loss_collect))}')
        #     print('Train Accuracy: {:.2f}% \n'.format(100*test_acc_collect[-1]))

        # Test inference after completion of training
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        test_acc_collect.append(test_acc)
        test_loss_collect.append(test_loss)

        print(
            f'\nResults after {epoch+1}/{args.epochs+1} global rounds of training:')
        print("Test Accuracy: {:.2f}%".format(100*test_acc))
        print(f'Test Loss    : {format(test_loss)}')

    # Saving the objects test_loss_collect and test_acc_collect:
    file_name = './save/objects/fedasync_{}_{}_{}_C{}_iid{}_E{}_B{}_Z{}_S{}_A{}_{}.pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs, args.byzantines, args.stale, args.alpha, time.time())

    with open(file_name, 'wb') as f:
        pickle.dump([test_loss_collect, test_acc_collect], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(test_loss_collect)), test_loss_collect, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/fedasync_{}_{}_{}_C{}_iid{}_E{}_B{}_Z{}_S{}_A{}_loss.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs, args.byzantines, args.stale, args.alpha))

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(test_acc_collect)), test_acc_collect, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/fedasync_{}_{}_{}_C{}_iid{}_E{}_B{}_Z{}_S{}_A{}_acc.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs, args.byzantines, args.stale, args.alpha))
