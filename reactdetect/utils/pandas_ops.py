"""
Common pandas operations
"""
import math
from collections import Counter

import pandas as pd 

from reactdetect.utils.magic_vars import PRIMARY_KEY_FIELDS
from reactdetect.utils.magic_vars import SUPPORTED_TARGET_MODELS

from tqdm import tqdm


def concat_multiple_df(list_of_dfs):
    return refresh_index(pd.concat(list_of_dfs, axis=0))


def refresh_index(df):
    return df.reset_index(drop=True)


def no_duplicate_index(df):
    return df.index.is_unique


def no_duplicate_perturbed_text(df, column_name):
    count = Counter(df[column_name])
    for sent in count.keys():
        if count[sent] > 1:
            return False
    return True


def no_duplicate_entry(df, column_name):
    return df[column_name].is_unique


# so that some legacy code still works...
def no_duplicate_preturbed_text(df, column_name):
    count = Counter(df[column_name])
    for sent in count.keys():
        if count[sent] > 1:
            return False
    return True


def drop_for_column_outside_of_values(df, column_name, values):
    return df[df[column_name].isin(values)]


def drop_for_column_inside_values(df, column_name, values):
    return df[~df[column_name].isin(values)]


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


def restrict_max_instance_for_class(in_df, attack_name_to_clip,
                                    max_instance_per_class=math.inf, min_instance_per_class=0,
                                    make_copy=False):
    """
    restrict the instances per class, class meaning attack_name
    """
    if make_copy:
        # make a copy of the dataset
        df = in_df.copy()
    else:
        df = in_df

    attack_names = list(set(sorted(df['attack_name'].unique())))  # duplicate class count ptsd

    # get data for each attack using the assigned samples
    dfs = []
    n_samples_per_class = []  # holds the number of samples for each attack
    for attack in attack_names:
        gf = df[(df['attack_name'] == attack)].copy()
        
        # downsample each class's dataframe untill add of the classes are <= maximum samples per class
        if len(gf) > max_instance_per_class and attack == attack_name_to_clip:
            gf = gf.head(max_instance_per_class)
        n_samples_per_class.append(len(gf))
        dfs.append(gf)
    
    df = pd.concat(dfs)
    df = refresh_index(df)
    assert no_duplicate_index(df)

    return df


def downsample_clean_to_max_nonclean_class(df):
    class_distribution = Counter(df['attack_name'])
    max_num_other_than_clean_per_class = sorted([i[1] for i in class_distribution.items() if i[0]!='clean'])[-1]
    return restrict_max_instance_for_class(df, 'clean', max_num_other_than_clean_per_class)


def downsample_clean_to_sum_nonclean_class(df):
    class_distribution = Counter(df['attack_name'])
    max_num_other_than_clean_per_class = sum([i[1] for i in class_distribution.items() if i[0]!='clean'])
    return restrict_max_instance_for_class(df, 'clean', max_num_other_than_clean_per_class)


def mask_df_target_model_of_clean_to(df, target_model_class):
    
    assert target_model_class in SUPPORTED_TARGET_MODELS
    assert no_duplicate_index(df)

    for idx in df.index:
        if df.at[idx, 'attack_name'] == 'clean':
            df.at[idx, 'target_model'] = target_model_class 

    return df 


def get_pk_tuple_from_pandas_row(pandas_row):
    return tuple([pandas_row[_PK] for _PK in PRIMARY_KEY_FIELDS])


def get_src_instance_identifier_from_pandas_row(pandas_row):
    """
    returns an tuple idenfier unique to src instance

    e.g. #777 sentence in SST is attacked by 7 attacks, those will share this identifier
    """
    return tuple([pandas_row['target_model_dataset'], pandas_row['test_index'], pandas_row['original_text_identifier']])


def split_df_by_column(idf, column):
    """
    https://stackoverflow.com/questions/40498463/python-splitting-dataframe-into-multiple-dataframes-based-on-column-values-and/40498517
    split a dataframe into sub dataframes, each grouped by unique values of that col
    return list of sub-dataframes
    """
    out = []
    for region, df_region in idf.groupby(column):
        out.append(df_region)
    return out


def convert_nested_list_to_df(df_list):
    """
    Converts a list of pd.DataFrame objects into one pd.DataFrame object.
    """
    return pd.concat(df_list)


def length_of_nested_list(list):
    count = 0
    for small_df in list:
        count += small_df.shape[0]
    return count

def generate_original_text_identifier(df):
    original_text_set = set(df['original_text'].unique())
    original_text_set_sorted = sorted(original_text_set)
    original_text_to_idx = dict(
        zip(
            original_text_set_sorted, 
            list(range(len(original_text_set_sorted)))
        )
    )
    df['original_text_identifier'] = df.apply(lambda row:original_text_to_idx[row['original_text']], axis=1)
    return df

def assert_original_text_identifier_is_unique(dat):
    """
    assert the field original_text_identifier has following behaviour:
        (1): could be used to split df into groups
        (2): each group only has **1** original text
        (3): this original text will never appear in other group.
    """
    print('running safety assertion')
    seen_original_text = set()
    seen_original_text_identifier = set()
    for g, data in tqdm(dat.groupby('original_text_identifier')):

        assert len(set(data['original_text_identifier'].unique())) == 1
        assert len(set(data['original_text'].unique())) == 1

        assert data['original_text'].unique()[0] not in seen_original_text
        seen_original_text.add(data['original_text'].unique()[0])

        assert data['original_text_identifier'].unique()[0] not in seen_original_text_identifier
        seen_original_text_identifier.add(data['original_text_identifier'].unique()[0])
