import math
import torch
import numpy as np
import scipy.sparse as sp
from itertools import groupby, chain
import globalvar as gol


def gen_adjacent_tab(adj):
    """
    generate adjacent table
    :param adj: <class 'torch.Tensor'> adjacent matrix
    :return adj_tab: <class 'dict'> adjaecnt table
    """
    adj_tab = {}
    adj_size = list(adj.size())[0]

    for i in range(adj_size):
        adj_info = adj[i]
        neigh_set = torch.nonzero(adj_info).numpy().flatten().tolist()
        adj_tab[str(i)] = set(neigh_set)

    return adj_tab


def gen_similarity_mx(adj_tab):
    """
    generate similarity matrix
    :param adj_tab: <class 'dict'> adjaecnt table
    :return adj_sim_mx: <class 'scipy.sparse.coo.coo_matrix'> similarity adjacent matrix
    """
    adj_tab_size = adj_tab.__len__()
    my_shape = (adj_tab_size, adj_tab_size)
    coo_mx = []
    for vtx_i in adj_tab:
        i = int(vtx_i)
        for vtx_j in adj_tab:
            j = int(vtx_j)
            similarity = (len(adj_tab[vtx_i] & adj_tab[vtx_j]) + 0.0) / len(adj_tab[vtx_i] | adj_tab[vtx_j])
            if i == j or similarity == 0:
                continue
            else:
                coo_mx.append([i, j, similarity])

    coo_mx = np.array(coo_mx)
    coo_mx = coo_mx[np.lexsort(-coo_mx.T)]
    _i = (coo_mx[:, 0]).astype(int)
    _j = (coo_mx[:, 1]).astype(int)
    _val = coo_mx[:, 2]

    adj_sim_mx = sp.coo_matrix((_val, (_i, _j)), shape=my_shape)
    return adj_sim_mx


def gen_compute_seq(adj):
    """
    generate vertexs' compute sequence
    :param adj: <class 'torch.Tensor'> adjacent matrix
    :return com_seq: <class 'numpy.ndarray'> compute sequence
    """
    com_seq = []

    # data preparing
    adj_tab = gen_adjacent_tab(adj)
    sim_mat = gen_similarity_mx(adj_tab)
    _dim = sim_mat.get_shape()[0]
    cnt = 0
    last_ele = None

    print('BEGIN!\nvtx remains:', sim_mat.getnnz())
    while sim_mat.getnnz() > 0:
        if last_ele is None:
            com_seq.append(',')
            idx = sim_mat.argmax()
            _i = math.floor(idx / _dim)
            _j = idx % _dim
            last_ele = (_i, _j)
        else:
            _i = last_ele[1]
            neigh_tensor = sim_mat.getcol(_i)
            nnz_nu = neigh_tensor.getnnz()
            if nnz_nu > 0:
                _j = neigh_tensor.argmax()
                last_ele = (_i, _j)
            else:
                last_ele = None

        sim_mat = delete_coo_i(sim_mat, _i)
        com_seq.append(_i)
        print('Round:', cnt, 'vtx remains:', sim_mat.getnnz(), ' this_vtx:', _i)
        cnt += 1

    # append the rest single vertex
    for i in range(_dim):
        if i not in com_seq:
            com_seq.append(i)
            com_seq.append(',')

    com_seq = unit_allocate(com_seq)
    com_seq = np.array(com_seq)
    return com_seq


def delete_coo_i(coo_mx, i):
    """
    delete pointed row and column
    :param coo_mx: <class 'scipy.sparse.coo.coo_matrix'> the target COO matrix
    :param i: <class 'int'> the row and column index to be deleted
    :return out: <class 'scipy.sparse.coo.coo_matrix'> the result COO matrix
    """
    # data preparing
    _col = coo_mx.col
    _row = coo_mx.row
    _data = coo_mx.data
    _shape = coo_mx.shape  # save the origin shape

    # calculate the delete index
    col_idx = set(np.where(_col==i)[0].tolist())
    row_idx = set(np.where(_row==i)[0].tolist())
    idx = list(col_idx|row_idx)

    # delete
    _col = np.delete(_col, idx)
    _row = np.delete(_row, idx)
    _data = np.delete(_data, idx)

    # reconstruct the COO matrix
    out = sp.coo_matrix((_data, (_row, _col)), shape=_shape)
    return out


def unit_allocate(com_seq):
    """

    :param com_seq:
    :return:
    """
    aggr_units_num = gol.get_value('aggr_units_num')
    com_seq = [list(g) for k, g in groupby(com_seq, lambda x: x == ',') if not k]
    com_seq.sort(key=lambda x: len(x) % aggr_units_num, reverse=True)
    temp_result = []
    for i in range(aggr_units_num):
        temp_result.append(
            [list(g) for k, g in groupby(com_seq, lambda x: len(x) % aggr_units_num) if k == i])

    a = []
    for i in range(aggr_units_num):
        if len(temp_result[i]) == 0:
            continue
        else:
            temp = temp_result[i][0]
        if i == 0:
            for j in temp:
                a += j
            temp_result[i][0] = []
        elif i % 2 == 0 and i == aggr_units_num / 2:
            for j in temp:
                a += j
            temp_result[i][0] = []
        else:
            aot_temp = temp_result[aggr_units_num-i][0]
            for j in temp:
                a += j
                if len(aot_temp) > 0:
                    k = aot_temp[-1]
                    a += k
                    aot_temp.remove(k)

            temp_result[i][0] = []
            temp_result[aggr_units_num-i][0] = aot_temp

    return a


