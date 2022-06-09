
import joblib
import os
import pandas as pd
import sys
import argparse

from collections import Counter

from scipy.__config__ import show

from reactdetect.utils.file_io import mkdir_if_dne

from reactdetect.utils.pandas_ops import drop_for_column_inside_values
from reactdetect.utils.pandas_ops import drop_for_column_outside_of_values
from reactdetect.utils.pandas_ops import assert_original_text_identifier_is_unique
from reactdetect.utils.pandas_ops import no_duplicate_index
from reactdetect.utils.pandas_ops import no_duplicate_entry
from reactdetect.utils.pandas_ops import generate_original_text_identifier
from reactdetect.utils.pandas_ops import get_pk_tuple_from_pandas_row
from reactdetect.utils.pandas_ops import get_src_instance_identifier_from_pandas_row
from reactdetect.utils.pandas_ops import split_df_by_column
from reactdetect.utils.pandas_ops import convert_nested_list_to_df

from reactdetect.utils.magic_vars import SUPPORTED_ATTACKS
from reactdetect.utils.magic_vars import SUPPORTED_ATTACK_VARIANTS

from tqdm import tqdm

RANDOM_SEED = 22
SUPPORTED_DATASETS = ['hatebase', 'civil_comments', 'wikipedia', 'sst', 'imdb', 'climate-change_waterloo', 'gab_dataset', 'reddit_dataset', 'wikipedia_personal']
SUPPORTED_TARGET_MODELS = ['bert', 'roberta', 'xlnet']

def show_df_stats(df):
    out = ''
    out += 'total_instances: '+str(len(df))+', \n'
    out += 'attack_name: '+str(dict(Counter(df['attack_name'])))+', \n'
    out += 'target_model_dataset: '+str(dict(Counter(df['target_model_dataset'])))+', \n'
    out += 'target_model: '+str(dict(Counter(df['target_model'])))+', \n'
    out += 'status: '+str(dict(Counter(df['status'])))+', \n'
    out += 'attack_toolchain: '+str(dict(Counter(df['attack_toolchain'])))+', \n'
    out += 'scenario: '+str(dict(Counter(df['scenario'])))+', \n'
    return out

def get_dataset_df(idf_dir='./whole_catted_dataset.csv'):
    """
    Read in the whole_catted_dataset.csv. Do some sanity check on it as well
    Pad useful column called pk, and another src instance identifier
    """
    odf = pd.read_csv(idf_dir)
    print('--- dropping unsupported datasets')
    odf = drop_for_column_outside_of_values(odf, 'target_model_dataset', SUPPORTED_DATASETS)
    print('--- dropping attack variants ')
    odf = drop_for_column_inside_values(odf, 'attack_name', SUPPORTED_ATTACK_VARIANTS)
    odf = odf.sample(frac=1, random_state=RANDOM_SEED)
    odf = generate_original_text_identifier(odf)
    odf['pk'] = odf.apply(lambda row: get_pk_tuple_from_pandas_row(row), axis=1)
    odf['unique_src_instance_identifier'] = odf.apply(lambda row: get_src_instance_identifier_from_pandas_row(row), axis=1)      
    assert no_duplicate_index(odf)
    assert no_duplicate_entry(odf, 'pk')
    assert_original_text_identifier_is_unique(odf)
    return odf

def create_ideal_train_test_split(df, split_ratio=0.6):
    src_column_name = 'unique_src_instance_identifier'
    idf_groups = split_df_by_column(df, src_column_name)
    train_groups = []
    test_groups = []
    len_train = 0
    len_test = 0

    for small_df in tqdm(idf_groups):
        if len_train==0 or len_train/(len_train + len_test)<split_ratio:
            train_groups.append(small_df)
            len_train += len(small_df)
        else:
            test_groups.append(small_df)
            len_test += len(small_df)

    train_df = convert_nested_list_to_df(train_groups)
    test_df = convert_nested_list_to_df(test_groups)
    s1 = set(train_df['unique_src_instance_identifier'])
    s2 = set(test_df['unique_src_instance_identifier'])
    assert len(s1.intersection(s2)) == 0, 'there duplicate entries across splits!'
    return train_df, test_df


def get_splits_for_dataset(dataset, df):
    print('--- filtering for dataset ', df.shape)
    df = drop_for_column_outside_of_values(df, 'target_model_dataset', [dataset])
    train_df, test_val_df = create_ideal_train_test_split(df, split_ratio=0.6)
    val_df, test_df = create_ideal_train_test_split(test_val_df, split_ratio=0.5)
    print('--- Train DF stats ---')
    print(show_df_stats(train_df))
    print('--- Test DF stats ---')
    print(show_df_stats(test_df))
    print('--- Val DF stats ---')
    print(show_df_stats(val_df))

    if dataset=='wikipedia_personal':
        assert len(set(train_df['attack_name'])) >= len(set(val_df['attack_name'])), 'train_df has more attacks than val_df'
        assert set(train_df['attack_name']).union(set(val_df['attack_name'])) == set(train_df['attack_name']), 'attacks missing in either of the splits'
        assert len(set(train_df['attack_name'])) >= len(set(test_df['attack_name'])), 'train_df has more attacks than test_df'
        assert set(train_df['attack_name']).union(set(test_df['attack_name'])) == set(train_df['attack_name']), 'attacks missing in either of the splits'
    else:       
        assert set(train_df['attack_name']) == set(test_df['attack_name']) == set(val_df['attack_name']), 'attacks missing in either of the splits'
    
    assert set(train_df['target_model_dataset']) == set(test_df['target_model_dataset']) == set(val_df['target_model_dataset']), 'discrepancy in target_model_dataset column across splits'
    assert set(train_df['target_model']) == set(test_df['target_model']) == set(val_df['target_model']), 'discrepancy in target_model distribution across splits'
    return train_df, val_df, test_df

def get_splits_for_tm(tm, train_df, val_df, test_df):
    print('--- filtering for target model ')
    train_df = drop_for_column_outside_of_values(train_df, 'target_model', [tm])
    val_df = drop_for_column_outside_of_values(val_df, 'target_model', [tm])
    test_df = drop_for_column_outside_of_values(test_df, 'target_model', [tm])
    print('--- Train DF stats ---')
    print(show_df_stats(train_df))
    print('--- Test DF stats ---')
    print(show_df_stats(test_df))
    print('--- Val DF stats ---')
    print(show_df_stats(val_df))

    assert set(train_df['target_model_dataset']) == set(test_df['target_model_dataset']) == set(val_df['target_model_dataset']), 'discrepancy in target_model_dataset column across splits'

    dataset = list(set(train_df['target_model_dataset']))[0]
    if dataset=='wikipedia_personal':
        assert len(set(train_df['attack_name'])) >= len(set(val_df['attack_name'])), 'train_df has less attacks than val_df'
        assert set(train_df['attack_name']).union(set(val_df['attack_name'])) == set(train_df['attack_name']), 'attacks missing in either of the splits'
        assert len(set(train_df['attack_name'])) >= len(set(test_df['attack_name'])), 'train_df has more attacks than test_df'
        assert set(train_df['attack_name']).union(set(test_df['attack_name'])) == set(train_df['attack_name']), 'attacks missing in either of the splits'
    else:       
        assert set(train_df['attack_name']) == set(test_df['attack_name']) == set(val_df['attack_name']), 'attacks missing in either of the splits'

    assert set(train_df['target_model']) == set(test_df['target_model']) == set(val_df['target_model']), 'discrepancy in target_model distribution across splits'
    return train_df, val_df, test_df

def create_splitted_whole_catted_dataset():
    print('--- reading data')
    df = get_dataset_df('./whole_catted_dataset.csv')
    try:
        df.drop(['Unnamed: 0', 'Unnamed: 0.1'], inplace=True, axis=1)
    except:
        pass
    print('--- dropping duplicates')
    df = df.drop_duplicates(subset=['target_model', 'target_model_dataset', 'attack_name', 'original_text_identifier'])

    df = df[~(df['target_model'] == 'uclmr')]
    assert 'uclmr' not in df['target_model'].unique()

    train_list = []
    val_list = []
    test_list = []

    print('--- making splits across all datasets ---')

    combined_dump_path = os.path.join(os.getcwd(), 'official_TCAB_splits', 'combined')
    mkdir_if_dne(combined_dump_path)

    for dataset in SUPPORTED_DATASETS:
        print(dataset)
        train_df_temp, val_df_temp, test_df_temp = get_splits_for_dataset(dataset, df)
        dataset_dump_path = os.path.join(os.getcwd(), 'official_TCAB_splits', 'splits_by_dataset', dataset)
        mkdir_if_dne(dataset_dump_path)
        for tm in SUPPORTED_TARGET_MODELS:
            train_df_temp_, val_df_temp_, test_df_temp_ = get_splits_for_tm(tm, train_df_temp, val_df_temp, test_df_temp)
            dataset_and_tm_dump_path = os.path.join(os.getcwd(), 'official_TCAB_splits', 'splits_by_dataset', dataset, tm)
            mkdir_if_dne(dataset_and_tm_dump_path)
            s1 = set(train_df_temp_['unique_src_instance_identifier'])
            s2 = set(val_df_temp_['unique_src_instance_identifier'])
            s3 = set(test_df_temp_['unique_src_instance_identifier'])
            assert len(s1.intersection(s2)) == 0
            assert len(s1.intersection(s3)) == 0
            assert len(s2.intersection(s3)) == 0
            train_df_temp_.to_csv(os.path.join(dataset_and_tm_dump_path, 'train.csv'))
            val_df_temp_.to_csv(os.path.join(dataset_and_tm_dump_path, 'val.csv'))
            test_df_temp_.to_csv(os.path.join(dataset_and_tm_dump_path, 'test.csv'))
            
            print('--- ', dataset, '', tm, ' --- DONE.')

        s1 = set(train_df_temp['unique_src_instance_identifier'])
        s2 = set(val_df_temp['unique_src_instance_identifier'])
        s3 = set(test_df_temp['unique_src_instance_identifier'])
        assert len(s1.intersection(s2)) == 0
        assert len(s1.intersection(s3)) == 0
        assert len(s2.intersection(s3)) == 0

        train_df_temp.to_csv(os.path.join(dataset_dump_path, 'train.csv'))
        val_df_temp.to_csv(os.path.join(dataset_dump_path, 'val.csv'))
        test_df_temp.to_csv(os.path.join(dataset_dump_path, 'test.csv'))


        print('--- ', dataset, ' --- DONE.')

        train_list.append(train_df_temp)
        val_list.append(val_df_temp)
        test_list.append(test_df_temp)

    print('--- combining splits across all datasets ---')
    
    train_df = convert_nested_list_to_df(train_list)
    val_df = convert_nested_list_to_df(val_list)
    test_df = convert_nested_list_to_df(test_list)


    s1 = set(train_df['unique_src_instance_identifier'])
    s2 = set(val_df['unique_src_instance_identifier'])
    s3 = set(test_df['unique_src_instance_identifier'])
    assert len(s1.intersection(s2)) == 0
    assert len(s1.intersection(s3)) == 0
    assert len(s2.intersection(s3)) == 0

    train_df.to_csv(os.path.join(combined_dump_path, 'train.csv'))
    test_df.to_csv(os.path.join(combined_dump_path, 'test.csv'))
    val_df.to_csv(os.path.join(combined_dump_path, 'val.csv'))

    print('--- DONE.')


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='hatebase', help='same as dir. in dropbox')
    # args = parser.parse_args()
    create_splitted_whole_catted_dataset()