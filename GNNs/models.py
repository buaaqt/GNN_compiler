import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers import GraphConvolution, MyGraphConvolution, GraphAttentionLayer, MyGraphAttentionLayer
from opts import Drop, Cat, LogSoftmax, ELu, ReLu, SaveFeat


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class MyGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(MyGCN, self).__init__()

        self.gc1 = MyGraphConvolution(nfeat, nhid, 0)
        self.gc2 = MyGraphConvolution(nhid, nclass, 1)
        self.dropout = dropout

    def forward(self, x, adj, opt_dic=None):
        x, f = self.gc1(x, adj, opt_dic)
        x = my_relu(x, opt_dic)
        x = my_dropout(x, self.dropout, training=self.training, op_dic=opt_dic)
        x, f = self.gc2(x, adj, opt_dic)
        x = my_log_softmax(x, opt_dic)
        return x

    def setparams(self, w1, w2):
        self.gc1.setparams(w1)
        self.gc2.setparams(w2)


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """
        Dense version of GAT.
        :param nfeat: 输入特征的维度
        :param nhid:  输出特征的维度
        :param nclass: 分类个数
        :param dropout: dropout
        :param alpha: LeakyRelu中的参数
        :param nheads: 多头注意力机制的个数
        """
        super(GAT, self).__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass
        self.dropout = dropout
        self.alpha = alpha
        self.nheads = nheads

        self.attentions = \
            [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

    def getinput(self):
        return self.nfeat, self.nhid, self.nclass, self.dropout, self.alpha, self.nheads


class MyGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(MyGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [MyGraphAttentionLayer(nfeat, nhid, dropout=dropout,
                                                 alpha=alpha, layer_id=int(_), concat=True)
                           for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = MyGraphAttentionLayer(nhid * nheads, nclass, dropout=dropout,
                                             alpha=alpha, layer_id=int(nheads), concat=False)

    def forward(self, x, adj, op_dic=None):
        x = my_dropout(x, self.dropout, training=self.training, op_dic=op_dic)
        x_list = []
        f_list = []
        for att in self.attentions:
            temp_x, temp_f = att(x, adj, op_dic)
            x_list.append(temp_x)
            f_list.append(temp_f)
        x = my_cat(f_list, x_list, op_dic)
        x = my_dropout(x, self.dropout, training=self.training, op_dic=op_dic)
        x, f = self.out_att(x, adj, op_dic, True)
        x = my_elu(x, op_dic)
        x = my_log_softmax(x, op_dic)
        return x


def my_cat(files_list, x_list, op_dic):
    x = torch.cat(x_list, dim=1)
    if op_dic is not None:
        # np_arr = x.detach().numpy()
        # np.savetxt('./output/' + op_dic['features_file'][0], np_arr, delimiter=",")
        op_dic['ops'].append(Cat(files_list, dim=1).to_dict())
    return x


def my_elu(x, op_dic):
    x = F.elu(x)
    if op_dic is not None:
        # np_arr = x.detach().numpy()
        # np.savetxt('./output/' + op_dic['features_file'][0], np_arr, delimiter=",")
        op_dic['ops'].append(ELu().to_dict())
    return x


def my_relu(x, op_dic):
    x = F.relu(x)
    if op_dic is not None:
        # np_arr = x.detach().numpy()
        # np.savetxt('./output/' + op_dic['features_file'][0], np_arr, delimiter=",")
        op_dic['ops'].append(ReLu().to_dict())
    return x


def my_dropout(x, dropout_rate, training, op_dic):
    x = F.dropout(x, dropout_rate, training=training)
    if op_dic is not None:
        # np_arr = x.detach().numpy()
        # np.savetxt('./output/' + op_dic['features_file'][0], np_arr, delimiter=",")
        op_dic['ops'].append(Drop(dropout_rate).to_dict())
    return x


def my_log_softmax(x, op_dic):
    x = F.log_softmax(x, dim=1)
    if op_dic is not None:
        # np_arr = x.detach().numpy()
        # np.savetxt('./output/' + op_dic['features_file'][0], np_arr, delimiter=",")
        op_dic['ops'].append(LogSoftmax(dim=1).to_dict())
    return x
