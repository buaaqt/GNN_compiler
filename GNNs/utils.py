import sys
import numpy as np
import scipy.sparse as sp
import torch
import pickle as pkl
import networkx as nx

def _parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def _pickle_load(pkl_file):
    if sys.version_info > (3, 0):
        return pkl.load(pkl_file, encoding='latin1')
    else:
        return pkl.load(pkl_file)


def encode_onehot(labels):
    classes = set(labels)
    temp_array = list(classes)
    temp_array.sort()
    classes_dict = {c: np.identity(len(temp_array))[i, :] for i, c in
                    enumerate(temp_array)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(dataset):
    if dataset is 'cora':
        return load_cora()
    elif dataset is 'dummy':
        return load_dummy()
    elif dataset is 'citeseer':
        return load_citeseer()


def load_cora(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    adj = torch.FloatTensor(np.array(adj.todense()))

    labels = torch.LongTensor(np.where(labels)[1])
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def load_dummy(path="../data/dummy/dummy_graph.npz", dataset="dummy"):
    """Load dummy dataset (a small test dataset made by ourselves)"""
    print('Loading {} dataset...'.format(dataset))

    r = np.load(path)
    adj = sp.coo_matrix(torch.from_numpy(r['Adj']))
    adj = normalize(adj + sp.eye(adj.shape[0]))
    features = normalize(r['feat'])
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.zeros(adj.shape[0], dtype=int))

    idx_train = torch.LongTensor(range(32))
    idx_val = torch.LongTensor(range(32))
    idx_test = torch.LongTensor(range(32))
    adj = torch.FloatTensor(np.array(adj.todense()))

    return adj, features, labels, idx_train, idx_val, idx_test


def load_citeseer(path="../data/citeseer", dataset="citeseer"):
    objnames = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(objnames)):
        with open("{}/ind.{}.{}".format(path, dataset, objnames[i]), 'rb') as f:
            objects.append(_pickle_load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = _parse_index_file("{}/ind.{}.test.index".format(path, dataset))
    test_idx_range = np.sort(test_idx_reorder)
    test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
    tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
    tx_extended[test_idx_range - min(test_idx_range), :] = tx
    tx = tx_extended
    ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
    ty_extended[test_idx_range - min(test_idx_range), :] = ty
    ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = features.tocsr()
    graph = nx.DiGraph(nx.from_dict_of_lists(graph))
    adj = nx.adjacency_matrix(graph)

    onehot_labels = np.vstack((ally, ty))
    onehot_labels[test_idx_reorder, :] = onehot_labels[test_idx_range, :]
    labels = np.argmax(onehot_labels, 1)

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    features = torch.FloatTensor(np.array(features.todense()))
    adj = torch.FloatTensor(np.array(adj.todense()))

    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test



