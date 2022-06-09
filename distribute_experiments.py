
import joblib
import os
import pandas as pd
from make_offical_dataset_splits import create_ideal_train_test_split

from reactdetect.utils.experiment import PresplittedExperimentNew
from reactdetect.utils.file_io import mkdir_if_dne
from reactdetect.utils.hashing import hash_old_pk_tuple
from reactdetect.utils.pandas_ops import downsample_clean_to_sum_nonclean_class
from reactdetect.utils.pandas_ops import downsample_clean_to_max_nonclean_class

OLD_PRIMARY_KEY_FIELDS = sorted(['scenario','target_model_dataset', 'target_model','attack_toolchain','attack_name', 'test_index'])

def get_splitted_exp(in_dir):
    print('Reading train.csv from ', os.path.join(in_dir, 'train.csv'))
    train_df = pd.read_csv(os.path.join(in_dir, 'train.csv'))
    print('Reading val.csv from ', os.path.join(in_dir, 'val.csv'))
    val_df = pd.read_csv(os.path.join(in_dir, 'val.csv'))
    print('Reading test.csv from ', os.path.join(in_dir, 'test.csv'))
    test_df = pd.read_csv(os.path.join(in_dir, 'test.csv'))
    return train_df, val_df, test_df

def drop_attacks_by_count(train_df, val_df, test_df, min_instance_per_class=10):
    """
    https://stackoverflow.com/questions/29403192/convert-series-returned-by-pandas-series-value-counts-to-a-dictionary
    """
    dropped_attacks = []
    counts = train_df['attack_name'].value_counts()
    drop_warning = ""
    for k in counts.to_dict().keys():
        if counts[k] < min_instance_per_class:
            drop_warning = drop_warning + k + ':' + str(counts[k]) + ', '
            dropped_attacks.append(k)
    if drop_warning != "":
        print(drop_warning, ' have been dropped due to having class count smaller than ', min_instance_per_class)
    train_df = train_df[~train_df['attack_name'].isin(counts[counts < min_instance_per_class].index)]
    val_df = val_df[~val_df['attack_name'].isin(dropped_attacks)]
    test_df = test_df[~test_df['attack_name'].isin(dropped_attacks)]
    return train_df, val_df, test_df

def add_target_label_column(df_row, experiment_setting):
    assert experiment_setting in ('clean_vs_all', 'multiclass_with_clean'), 'incorrect experiment setting'
    if experiment_setting=='clean_vs_all':
        if df_row['attack_name'] == 'clean':
            return 'clean'
        else:
            return 'perturbed'
    elif experiment_setting=='multiclass_with_clean':
        return df_row['attack_name']

def filter_by_exp_setting(train_df, val_df, test_df_release, test_df_hidden, experiment_setting):
    assert experiment_setting in ('clean_vs_all', 'multiclass_with_clean')
    # not touching test_df.csv as it's going to remain "hidden till the end"
    out = []
    train_df['target_label'] = train_df.apply(lambda row: add_target_label_column(row, experiment_setting), axis=1)
    val_df['target_label'] = val_df.apply(lambda row: add_target_label_column(row, experiment_setting), axis=1)
    test_df_release['target_label'] = test_df_release.apply(lambda row: add_target_label_column(row, experiment_setting), axis=1)
    test_df_hidden['target_label'] = test_df_hidden.apply(lambda row: add_target_label_column(row, experiment_setting), axis=1)
    if experiment_setting=='clean_vs_all':
        train_df = downsample_clean_to_sum_nonclean_class(train_df)
        val_df = downsample_clean_to_sum_nonclean_class(val_df)
        # test_df = downsample_clean_to_sum_nonclean_class(test_df)
        out.append(PresplittedExperimentNew(train_df=train_df, val_df=val_df, test_df_release=test_df_release, test_df_hidden=test_df_hidden, name='clean_vs_all'))
        return out
    
    elif experiment_setting=='multiclass_with_clean':
        train_df = downsample_clean_to_max_nonclean_class(train_df)
        val_df = downsample_clean_to_max_nonclean_class(val_df)
        # test_df = downsample_clean_to_max_nonclean_class(test_df)
        out.append(PresplittedExperimentNew(train_df=train_df, val_df=val_df, test_df_release=test_df_release, test_df_hidden=test_df_hidden, name='multiclass_with_clean'))
        return out

def check_joblib_dict(samples_dict):
    """
    Checks to make sure each example has the correct
        number of fields in the primary key; if not,
        this method attempts to fill in the missing values.

    Input
        samples_dict: dict of extracted features
            format: key - sample index, value - dict.

    Return
        samples_dict, with updated primary keys and unique IDs.
    """
    # actual primary key as len=7 pk in job has len=6. we take care of this discrepancy in fn df_to_instance_subset
    for key, d in samples_dict.items():
        assert len(d['primary_key']) == 6
    return samples_dict

def wiki_civil_remove_unextracted_clean(train_df, val_df, test_df, args):

    print('--- dropping clean instances with unextracted features')
    # get ids of clean instances with extracted features
    sw_reprs_dir = os.path.join('reprs', 'samplewise')
    fp = os.path.join(sw_reprs_dir, f'{args.target_model}_{args.target_model_dataset}_clean_.joblib')
    sw = joblib.load(fp)
    sw = check_joblib_dict(sw)  # fills in missing primary key fields
    ids_with_features = [sw[k]['unique_id'] for k in sw.keys()]

    # get unique ids for current dataframe
    pk_list_train = train_df[OLD_PRIMARY_KEY_FIELDS].itertuples(index=False)
    pk_list_val = val_df[OLD_PRIMARY_KEY_FIELDS].itertuples(index=False)
    pk_list_test = test_df[OLD_PRIMARY_KEY_FIELDS].itertuples(index=False)

    train_df['id'] = [hash_old_pk_tuple(tuple(tup)) for tup in pk_list_train]
    val_df['id'] = [hash_old_pk_tuple(tuple(tup)) for tup in pk_list_val]
    test_df['id'] = [hash_old_pk_tuple(tuple(tup)) for tup in pk_list_test]

    # partition clean from all other attacks
    not_clean_df_train = train_df[train_df['attack_name'] != 'clean']
    not_clean_df_val = val_df[val_df['attack_name'] != 'clean']
    not_clean_df_test = test_df[test_df['attack_name'] != 'clean']

    # keep clean instances that have extracted features
    clean_df_train = train_df[train_df['attack_name'] == 'clean']
    clean_df_train = clean_df_train[clean_df_train['id'].isin(ids_with_features)]
    clean_df_val = val_df[val_df['attack_name'] == 'clean']
    clean_df_val = clean_df_val[clean_df_val['id'].isin(ids_with_features)]
    clean_df_test = test_df[test_df['attack_name'] == 'clean']
    clean_df_test = clean_df_test[clean_df_test['id'].isin(ids_with_features)]

    # put the partitions back together
    train_df = pd.concat([not_clean_df_train, clean_df_train])
    val_df = pd.concat([not_clean_df_val, clean_df_val])
    test_df = pd.concat([not_clean_df_test, clean_df_test])
    del train_df['id']
    del val_df['id']
    del test_df['id']

    return train_df, val_df, test_df

def main(args):
    try:
        in_dir = os.path.join(os.getcwd(), 'official_TCAB_splits', 'splits_by_dataset_and_tm', args.target_model_dataset, args.target_model)
        assert os.path.exists(os.path.join(in_dir, 'train.csv')), 'no train.csv in in_dir, get splits from google drive'
        assert os.path.exists(os.path.join(in_dir, 'val.csv')), 'no val.csv in in_dir, get splits from google drive'
        assert os.path.exists(os.path.join(in_dir, 'test.csv')), 'no test.csv in in_dir, get splits from google drive'
    except AssertionError as e:
        print(e)

    out_dir = os.path.join(os.path.join('detection-experiments', args.target_model_dataset, args.experiment_setting, args.target_model))
    train_df, val_df, test_df = get_splitted_exp(in_dir)

    if args.target_model_dataset in ('civil_comments', 'wikipedia'):
        train_df, val_df, test_df = wiki_civil_remove_unextracted_clean(train_df, val_df, test_df, args)

    assert set(train_df['attack_name']) == set(val_df['attack_name']) == set(test_df['attack_name']), 'attacks missing in train, val or test split'
 
    print('Dropping attacks by count, if less than < 10')
    train_df, val_df, test_df = drop_attacks_by_count(train_df, val_df, test_df)
    test_df_release, test_df_hidden = create_ideal_train_test_split(test_df, split_ratio=0.5)
    
    for e in filter_by_exp_setting(train_df, val_df, test_df_release, test_df_hidden, args.experiment_setting):
        e.aux = vars(args)
        e.dump(exp_root_dir=out_dir)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--target_model_dataset', type=str, default='hatebase', help='same as dir. in dropbox')
    parser.add_argument('--target_model', type=str, default='xlnet',
                        help='same as dir. in dropbox, can be combo like bert+roberta')
    parser.add_argument('--experiment_setting', type=str, default='clean_vs_all')    
    args = parser.parse_args()
    main(args)
