import globalvar as gol


class Aggr:
    """generate aggregate operation's dictionary"""
    def __init__(self, idx, nei, weights, dst, reuse_set=[], consist=True):
        self.aggr_idx = idx
        self.reuse_set = reuse_set
        self.nei = nei
        self.neigh_w = weights
        self.dst = dst
        self.consist = consist

    def to_dict(self):
        aggr_units_num = gol.get_value('aggr_units_num')
        load_set = list(set(self.nei)-set(self.reuse_set))
        reu_set = list(set(self.reuse_set) & set(self.nei))

        if len(reu_set) > 0:
            idx = [self.nei.index(i) for i in reu_set]
            reu_set_wei = [self.neigh_w[_idx] for _idx in idx]
        else:
            reu_set_wei = []

        if len(load_set) > 0:
            idx = [self.nei.index(i) for i in load_set]
            load_set_wei = [self.neigh_w[_idx] for _idx in idx]
        else:
            load_set_wei = []
        return {
            'aggr_idx': self.aggr_idx,
            'unit_idx': self.aggr_idx % aggr_units_num,
            'opt': 'AGGR',
            'neighbors': self.nei,
            'reuse_set': reu_set,
            'load_set': load_set,
            'consistence_flag': self.consist,
            'neigh_weights': self.neigh_w,
            'reuse_set_wei': reu_set_wei,
            'load_set_wei': load_set_wei,
            'dst': self.dst
        }


class MulAdd:
    """generate multiple and add operation's dictionary"""
    def __init__(self, row_range, col_range, has_bias_flag):
        self.row_r = row_range
        self.col_r = col_range
        self.bias_f = has_bias_flag

    def to_dict(self):
        return {
            'opt': 'MUL_ADD',
            'bias_flag': self.bias_f,
            'row_range': self.row_r,
            'col_range': self.col_r,
        }


class SaveFeat:
    """generate save features operation's dictionary"""
    def __init__(self, layer_id):
        self.layer_id = layer_id

    def to_dict(self):
        return {
            'opt': 'SAVE_FEAT',
            'file': 'LAYER_' + str(self.layer_id) + '_OUT_FEAT.csv'
        }


class SaveLayerInfo:
    """generate save every single layer's information dictionary"""
    def __init__(self, layer_id, f_name):
        self.layer_id = layer_id
        self.f_name = f_name

    def to_dict(self):
        return {
            'opt': 'LAYER_INFO',
            'lay_id': self.layer_id,
            'file': self.f_name
        }


class Drop:
    """generate dropout operation's dictionary"""
    def __init__(self, dropout_rate):
        self.drop = dropout_rate

    def to_dict(self):
        return {
            'opt': 'DROPOUT',
            'drop_rate': self.drop
        }


class ELu:
    """generate elu operation's dictionary"""
    def __init__(self):
        self.name = 'ELU'

    def to_dict(self):
        return {
            'opt': self.name
        }


class ReLu:
    """generate elu operation's dictionary"""
    def __init__(self):
        self.name = 'ReLU'

    def to_dict(self):
        return {
            'opt': self.name
        }


class LogSoftmax:
    """generate log_softmax operation's dictionary"""
    def __init__(self, dim=1):
        self.dim = dim

    def to_dict(self):
        return {
            'opt': 'LOGSOFTMAX',
            'dim': self.dim
        }


class Cat:
    """generate cat operation's dictionary"""
    def __init__(self, file_list, dim=1):
        self.file_list = file_list
        self.dim = dim

    def to_dict(self):
        return {
            'opt': 'CAT',
            'files_list': self.file_list,
            'dim': self.dim
        }


