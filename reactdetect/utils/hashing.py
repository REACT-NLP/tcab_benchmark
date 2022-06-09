import hashlib


def hash_pk_tuple(tup):
    assert(len(tup) == 7), 'react pk does not have 7 items!'
    str_tup = '_'.join(str(i) for i in tup)
    id_ = str(int(hashlib.md5(str_tup.encode('ascii')).hexdigest(), 16))
    return id_

def hash_old_pk_tuple(tup):
    assert(len(tup) == 6), 'react pk does not have 6 items!'
    str_tup = '_'.join(str(i) for i in tup)
    id_ = str(int(hashlib.md5(str_tup.encode('ascii')).hexdigest(), 16))
    return id_


def get_pk_tuple_old(df, index):
    _PK = sorted(['scenario', 'target_model_dataset', 'target_model',
                  'attack_toolchain', 'attack_name', 'test_index'])
    _PK = [df.at[index, i] for i in _PK]
    return _PK


def get_pk_tuple(df, index):
    _PK = sorted(['scenario', 'target_model_dataset', 'target_model',
                  'attack_toolchain', 'attack_name', 'test_index',
                  'original_text_identifier'])
    _PK = [df.at[index, i] for i in _PK]
    return _PK
