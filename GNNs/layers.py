import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from opts import Aggr, MulAdd, SaveFeat, SaveLayerInfo
import globalvar as gol


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return '\n' \
               + self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')\n' \
               + 'weight:' + str(self.weight) + ',\n' \
               + 'bias:' + str(self.bias) + ',\n'


def multiply_add(weights, x, bias, op_dic):
    """
    multiply and then add the bias, considering the systolic array size
    partition the result matrix as [dim x dim]
    compute a single block every time
    """
    # data preparing
    systolic_array_dim = 16
    result_mx_rows = x.size()[0]-1
    result_mx_cols = weights.size()[1]-1

    # blocked partitioning
    temp_cols = result_mx_cols
    while temp_cols > 0:
        col_range = get_range(temp_cols, systolic_array_dim)
        cols = weights[col_range[0]:col_range[1]+1, :]
        if bias is not None:
            bias_partitioned = bias[col_range[0]:col_range[1]+1]
            bias_partitioned = bias_partitioned.data.numpy().tolist()
        else:
            bias_partitioned = None

        temp_rows = result_mx_rows
        while temp_rows > 0:
            row_range = get_range(temp_rows,systolic_array_dim)
            rows = x[row_range[0]:row_range[1]+1, :]
            cols_back = cols.data.numpy().tolist()
            rows_back = rows.data.numpy().tolist()

            # dump to file
            if op_dic is not None:
                if bias is not None:
                    op_dic['ops'].append(MulAdd(row_range, col_range, True).to_dict())
                else:
                    op_dic['ops'].append(MulAdd(row_range, col_range, False).to_dict())

            temp_rows = temp_rows-systolic_array_dim
        temp_cols = temp_cols-systolic_array_dim

    if bias is not None:
        return torch.mm(x, weights) + bias
    else:
        return torch.mm(x, weights)


def get_range(x, dim):
    """simple partition function"""
    if x > dim:
        return [x-dim+1, x]
    else:
        return [0, x]


class MyGraphConvolution(Module):
    """Native GCN layer"""
    def __init__(self, in_features, out_features, layer_id, bias=False):
        super(MyGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layer_id = layer_id
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def setparams(self, w):
        self.weight = Parameter(torch.from_numpy(w).float())
        w_shape = self.weight.size()
        self.in_features = w_shape[0]
        self.out_features = w_shape[1]
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, op_dic):
        aggr_units_num = gol.get_value('aggr_units_num')
        adj_size = list(adj.size())[0]
        update_list = [0] * adj_size

        # save layer information to files
        if op_dic is not None:
            temp_bias = None
            if self.bias is not None:
                temp_bias = self.bias
            layer_info = {
                'weights': self.weight.tolist(),
                'bias': temp_bias
            }
            file_name = './output/layer_' + str(self.layer_id) + '_info.json'
            with open(file_name, 'w') as lay_f:
                json.dump(layer_info, lay_f, indent=4)
            op_dic['ops'].append(SaveLayerInfo(self.layer_id, file_name).to_dict())

        # update
        output = multiply_add(self.weight, input, self.bias, op_dic)

        # load the pre-computed computing sequence
        com_seq = (np.loadtxt('./loads/compute_seq.csv')).astype(int)

        for i in com_seq:
            i = int(i)
            adj_info = adj[i]
            nei_list = torch.nonzero(adj_info).numpy().flatten().tolist()
            aggr_num = len(nei_list)

            feat_sum = (output[nei_list]).sum(0)
            _feat = feat_sum / aggr_num
            update_list[i] = _feat

            if op_dic is not None:
                if len(op_dic['ops']) > 0 and op_dic['ops'][-1]['opt'] == 'AGGR':
                    last_aggr = op_dic['ops'][-1]
                    last_aggr_ldx = last_aggr['aggr_idx']
                    last_aggr_neigh = last_aggr['neighbors']
                    unit_idx = (last_aggr_ldx + 1) % aggr_units_num
                    if unit_idx == 0:
                        last_aggr_neigh = []
                    op_dic['ops'].append(
                        Aggr(last_aggr_ldx+1, nei_list, [1./aggr_num]*aggr_num,
                             i, reuse_set=last_aggr_neigh).to_dict())
                else:
                    op_dic['ops'].append(
                        Aggr(0, nei_list, [1./aggr_num]*aggr_num, i).to_dict())

        output = torch.stack(update_list, dim=0)

        # dump new feats to file
        file_name = 'LAYER_' + str(self.layer_id) + '_OUT_FEAT.csv'
        if op_dic is not None:
            # np_arr = output.detach().numpy()
            # np.savetxt('./output/' + file_name, np_arr, delimiter=",")
            pass

        return output, file_name

    def __repr__(self):
        return '\n' \
               + self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')\n' \
               + 'weight:' + str(self.weight) + ',\n' \
               + 'bias:' + str(self.bias) + ',\n'


class GraphAttentionLayer(Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)],
                            dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        # att = attention.data.numpy()
        # print(attention.size())
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + \
               ' -> ' + str(self.out_features) + ')'


class MyGraphAttentionLayer(Module):
    """Native Simple GAT layer"""

    def __init__(self, in_features, out_features, dropout, alpha, layer_id, concat=True):
        """
        :dropout:Dropout rate (1 - keep probability).
        :alpha: Alpha for the leaky_relu.
        """
        super(MyGraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.layer_id = layer_id

        self.W = Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.bias = None
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj, op_dic, out_flag=False):
        aggr_units_num = gol.get_value('aggr_units_num')
        # save layer information to files
        if op_dic is not None:
            layer_info = {
                'weights': self.W.tolist(),
                'bias': self.bias
            }
            file_name = './output/layer_' + str(self.layer_id) + '_info.json'
            with open(file_name, 'w') as lay_f:
                json.dump(layer_info, lay_f, indent=4)
            op_dic['ops'].append(SaveLayerInfo(self.layer_id, file_name).to_dict())

        h = multiply_add(self.W, input, self.bias, op_dic)
        N = h.size()[0]
        feat_dim = h.size()[1]
        my_shape = (adj.size()[0], adj.size()[1])

        # a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)],
        #                     dim=1).view(N, -1, 2 * self.out_features)
        # e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        #
        # zero_vec = -9e15*torch.ones_like(e)
        # attention = torch.where(adj > 0, e, zero_vec)
        # attention = F.softmax(attention, dim=1)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        _i = []
        _j = []
        _w = []

        com_seq = (np.loadtxt('./loads/compute_seq.csv')).astype(int)
        for i in com_seq:
            i = int(i)
            adj_info = adj[i]
            nei_list = torch.nonzero(adj_info).numpy().flatten().tolist()
            nei_len = len(nei_list)

            this_att = torch.cat([h[nei_list], h[i].view(1, feat_dim).repeat(nei_len, 1)], dim=1)
            this_att = self.leakyrelu(torch.matmul(this_att, self.a))
            this_att = F.softmax(this_att, dim=0)
            _i.extend([i] * nei_len)
            _j.extend(nei_list)
            _w.extend(this_att.squeeze(1).data.numpy().tolist())

            a_weights = this_att.data.numpy().flatten()

            if op_dic is not None:
                if len(op_dic['ops']) > 0 and op_dic['ops'][-1]['opt'] == 'AGGR':
                    last_aggr = op_dic['ops'][-1]
                    last_aggr_ldx = last_aggr['aggr_idx']
                    last_aggr_neigh = last_aggr['neighbors']
                    if (last_aggr_ldx + 1) % aggr_units_num == 0:
                        last_aggr_neigh = []
                    op_dic['ops'].append(
                        Aggr(last_aggr_ldx + 1, nei_list, a_weights.tolist(),
                             i, reuse_set=last_aggr_neigh, consist=False).to_dict())
                else:
                    op_dic['ops'].append(
                        Aggr(0, nei_list, a_weights.tolist(), i, consist=False).to_dict())

        attention = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix((_w, (_i, _j)), shape=my_shape))
        h_prime = torch.spmm(attention, h)

        if self.concat:
            output = F.elu(h_prime)
        else:
            output = h_prime

        # dump new feats to file
        file_name = 'LAYER_' + str(self.layer_id) + '_OUT_FEAT.csv'
        if op_dic is not None and out_flag is False:
            # np_arr = output.detach().numpy()
            # np.savetxt('./output/' + file_name, np_arr, delimiter=",")
            op_dic['ops'].append(SaveFeat(self.layer_id).to_dict())
            op_dic['features_file'].append(file_name)

        return output, file_name

    def setparams(self, W, a):
        self.a = a
        self.W = W

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + \
               ' -> ' + str(self.out_features) + ')'


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

