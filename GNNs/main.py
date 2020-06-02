from __future__ import division
from __future__ import print_function

import os
import glob
import time
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_data, accuracy
from models import GCN, MyGCN, GAT, MyGAT
from optimize import gen_compute_seq
from evaluate import evaluate
import globalvar as gol

# Global Variables
model = None
optimizer = None
args = None


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test(opt_dic=None, train_flag=False):
    model.eval()
    output = model(features, adj, opt_dic)
    # if the model is 'GCN' or 'GAT', undo this comment
    # output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    # save GNN model
    if train_flag:
        torch.save(model, "./loads/GNN_model")

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


if __name__ == "__main__":
    gol.init()
    gol.set_value('aggr_units_num', 4)
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--trainmode', action='store_true', default=False, help='Enable training.')
    parser.add_argument('--seqgenflag', action='store_true', default=False, help='Enable generating compute sequence.')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--dataset', type=str, default='citeseer', help='Select a dataset.')
    parser.add_argument('--model', type=str, default='GCN', help='Select GNN model.')
    # general paras
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    # GAT paras
    parser.add_argument('--nb_heads', type=int, default=2, help='Number of head attentions.')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leakyrelu.')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset)

    # Model and optimizer
    if args.model is 'GCN':
        print('Using model: GCN')
        model = MyGCN(nfeat=features.shape[1],
                      nhid=args.hidden,
                      nclass=labels.max().item() + 1,
                      dropout=args.dropout)
        if (args.trainmode is False) and (args.dataset is 'dummy'):
            r = np.load("../data/dummy/dummy_graph.npz")
            w1 = r['weight_layer1']
            w2 = r['weight_layer2']
            model.setparams(w1, w2)

    elif args.model is 'GAT':
        print('Using model: GAT')
        model = MyGAT(nfeat=features.shape[1],
                      nhid=args.hidden,
                      nclass=int(labels.max()) + 1,
                      dropout=args.dropout,
                      nheads=args.nb_heads,
                      alpha=args.alpha)

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    if args.seqgenflag:
        com_seq = gen_compute_seq(adj)
        np.savetxt('./loads/compute_seq.csv', com_seq, delimiter=",")

    if args.trainmode:
        # Train model
        t_total = time.time()
        for epoch in range(args.epochs):
            train(epoch)
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Testing
        test(train_flag=True)

    else:
        # Load model and dataset, then execute an inference, objected to architecture
        # Sum: Sum up neighbour vertex's features
        # Const_Mul: Multiply two constants or a constant and a var
        # Relu:
        # Mul_Add: Wx+b

        # init operation dictionary
        operation_dict = {'features_file': ['features_file.csv'],
                          'features_shape': None,
                          'ops': []
                          }

        if (args.model is 'GCN') and (args.trainmode is False) and (args.dataset is 'dummy'):
            pass
        else:
            model = torch.load("./loads/GNN_model")

        # save features, file-interaction with architecture
        if args.cuda:
            np_arr = features.cpu().numpy()
        else:
            np_arr = np.array(features)
        operation_dict['features_shape'] = np_arr.shape
        np.savetxt('./output/features_file.csv', np_arr, delimiter=",")

        with open('./output/opts.json', 'w') as opt_f:
            test(opt_dic=operation_dict, train_flag=False)
            json.dump(operation_dict, opt_f, indent=4)

        evaluate(operation_dict)

