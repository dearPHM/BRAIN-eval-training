#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import time
import pickle
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from utils import get_dataset
from options import args_parser
from update import test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar


if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()

    device = torch.device('cuda' if (
        args.gpu != None and torch.cuda.is_available()) else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(args.gpu)

    # load datasets
    train_dataset, test_dataset, _ = get_dataset(args)

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
    global_model = global_model.to(device)
    global_model.train()
    print(global_model)

    # Training
    # Set optimizer and criterion
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                    momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr,
                                     weight_decay=1e-4)

    # trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    trainloader = DataLoader(
        train_dataset, batch_size=args.local_bs, num_workers=4, shuffle=True)

    # criterion = torch.nn.NLLLoss().to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    test_loss_collect, test_acc_collect = [], []

    for epoch in tqdm(range(args.epochs)):
        # batch_loss = []
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = global_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if args.verbose and (batch_idx % 50 == 0):
                print('Train Epoch: {:4d} [{:6d}/{:6d} ({:3.0f}%)]\tLoss: {:10.6f}'.format(
                    epoch+1, batch_idx * len(images), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item()))

            # batch_loss.append(loss.item())

        # loss_avg = sum(batch_loss)/len(batch_loss)
        # print('\nTrain loss:', loss_avg)
        # test_loss_collect.append(loss_avg)

        # testing
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        test_acc_collect.append(test_acc)
        test_loss_collect.append(test_loss)
        # print('\nTest on', len(test_dataset), 'samples')
        # print("Test Accuracy: {:.2f}%".format(100*test_acc))
        # print(f'Test Loss    : {format(test_loss)}')

    # Saving the objects test_loss_collect and test_acc_collect:
    file_name = './save/objects/nn_{}_{}_{}_iid{}_{}.pkl'.\
        format(args.dataset, args.model, args.epochs, args.iid, time.time())

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
    plt.savefig('./save/nn_{}_{}_{}_iid{}_loss.png'.format(args.dataset, args.model,
                                                           args.epochs, args.iid))

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(test_acc_collect)), test_acc_collect, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/nn_{}_{}_{}_iid{}_acc.png'.format(args.dataset, args.model,
                                                          args.epochs, args.iid,))
