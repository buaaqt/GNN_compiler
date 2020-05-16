def evaluate(op_dic):
    total_time = 0
    memory_access_eval(op_dic)
    for op in op_dic['ops']:
        _type = op['opt']
        if _type is 'AGGR':
            total_time += aggr_time(op)
        elif _type is 'MUL_ADD':
            total_time += mul_time(op)
        elif _type is 'SAVE_FEAT':
            total_time += save_feat_time(op)
        elif _type is 'LAYER_INFO':
            total_time += save_layer_time(op)
        elif _type is 'DROPOUT':
            total_time += drop_time(op)
        elif _type is 'ELU':
            total_time += elu_time(op)
        elif _type is 'ReLU':
            total_time += relu_time(op)
        elif _type is 'LOGSOFTMAX':
            total_time += logsoftmax_time(op)
        elif _type is 'CAT':
            total_time += cat_time(op)
        else:
            pass

    return total_time


def aggr_time(op):
    reuse_coe = 0.1
    load_coe = 0.5
    aggr_coe = 0.1

    reuse_set = op['reuse_set']
    load_set = op['load_set']
    data_prep_time = reuse_coe*len(reuse_set) + load_coe*len(load_set)

    com_time = aggr_coe*(len(reuse_set) + len(load_set))

    return data_prep_time + com_time


def mul_time(op):
    mul_coe = 0.1
    row_dim = op['row_range'][1] - op['row_range'][0] + 1
    col_dim = op['col_range'][1] - op['col_range'][0] + 1

    return row_dim*col_dim*mul_coe


def save_feat_time(op):
    return 30


def save_layer_time(op):
    return 20


def drop_time(op):
    return 0.5


def elu_time(op):
    return 0.5


def relu_time(op):
    return 0.5


def logsoftmax_time(op):
    return 0.5


def cat_time(op):
    cat_len = len(op['files_list'])
    return 0.3*cat_len


def memory_access_eval(op_dic):
    memory_access_times = 0
    raw_access_times = 0
    for op in op_dic['ops']:
        if op['opt'] is 'AGGR':
            memory_access_times += len(op['load_set'])
            raw_access_times += len(op['neighbors'])
    print('Memory access times without optimizing: ', raw_access_times)
    print('Actual memory access times: ', memory_access_times)



