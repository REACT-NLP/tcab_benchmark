"""
Useful helpers for partitioning the data
"""
import joblib
import os
import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utility import convert_nested_list_to_df

def split_df_by_column(idf, column):
    """
    split dataframe by column
    """
    out = []
    for region, df_region in idf.groupby(column):
        out.append(df_region)
    return out

def drop_attacks_by_count(df, min_instance_per_class=10):
    """
        https://stackoverflow.com/questions/29403192/convert-\
        series-returned-by-pandas-series-value-counts-to-a-dictionary
    """
    counts = df['attack_name'].value_counts()
    drop_warning = ""
    for k in counts.to_dict().keys():
        if counts[k] < min_instance_per_class:
            drop_warning = drop_warning+k+':'+str(counts[k])+', '
    if drop_warning != "":
        print(
            drop_warning, 
            ' have been dropped due to having class count smaller than ', 
            min_instance_per_class
            )
    res = df[~df['attack_name'].isin(counts[counts < min_instance_per_class].index)]
    return res

def train_test_split_df(idf, test_size=0.2):
    """
    train_test_split, but never use same instance from src instance in both training and test.
        e.g. if sentence #777 from nuclear_energy is attacked by 3 atk aiming at 4 tgt mdl, 
        these should *all* go to either train or test set
        this makes fully balancing classes hard, we'll make use of random and just oversample later
    """
    assert isinstance(idf, pd.DataFrame)
    dfs_by_src_instance = drop_attacks_by_count(idf)
    dfs_by_src_instance = split_df_by_column(dfs_by_src_instance, 'unique_src_instance_identifier') 
    shuffle_sucessful = False
    for i in range(5):
        split_point =  int(len(dfs_by_src_instance)*(1-test_size))
        random.Random(i).shuffle(dfs_by_src_instance)
        train_df = pd.concat(dfs_by_src_instance[:split_point])
        test_df = pd.concat(dfs_by_src_instance[split_point:])
        if set(train_df['attack_name'].unique()) == set(test_df['attack_name'].unique()):
            shuffle_sucessful = True 
            break 
    if not shuffle_sucessful:
        print('could not get a train-test split with coherent label, tried 5 times')
        raise AssertionError
    
    return train_df, test_df

def oversample(df):
    """
    oversample dataframe.
    ** Assumes field label exists.
    """
    lst = [df]
    max_size = df['label'].value_counts().max()
    for class_index, group in df.groupby('label'):
        lst.append(group.sample(max_size-len(group), replace=True, random_state=0))
    frame_new = pd.concat(lst)
    return frame_new 

def prepare_df_for_bert_training(df, setting, label_encoder = None):
    """
    train.main assumes csv files with
    text and label fields.
    """
  
    if label_encoder == None:
        label_encoder = LabelEncoder().fit(df['target_label'])
    df['text'] = df['perturbed_text']
    df['label'] = label_encoder.transform(df['target_label'])
    
    return df, label_encoder

def main(data_root):
    """
    takes in a path containing train/test.csv
    create a subfolder called (detection)BERT
    dumps processed train/val/test csv inside.
    """
    assert {'train.csv', 'val.csv', 'test_release.csv'}.issubset(set(os.listdir(data_root)))
    assert 'clean_vs_all' in data_root or 'multiclass_with_clean' in data_root
    train_df_temp = pd.read_csv(
        os.path.join(data_root, 'train.csv')
    )
    # treat val split as test split
    test_df = pd.read_csv(
        os.path.join(data_root, 'val.csv')
    )
    # only for getting labels for leaderboard submission
    test_df_leaderboard = pd.read_csv(
        os.path.join(data_root, 'test_release.csv')
    )

    train_df, val_df = train_test_split_df(train_df_temp)

    setting = 'clean_vs_all' if 'clean_vs_all' in data_root else 'multiclass_with_clean'
    train_df, le  = prepare_df_for_bert_training(train_df, setting)
    val_df, _  = prepare_df_for_bert_training(val_df, setting, le)
    test_df, _  = prepare_df_for_bert_training(test_df, setting, le)
    test_df_leaderboard, _ = prepare_df_for_bert_training(test_df, setting, le)

    train_df = oversample(train_df)
    val_df = oversample(val_df)

    output_root = os.path.join(data_root, 'BERT_DETECTION')
    os.makedirs(
        output_root, 
        exist_ok=True
        )
    
    train_df.to_csv(
        os.path.join(output_root, 'train.csv')
        )
    val_df.to_csv(
        os.path.join(output_root, 'val.csv')
        )
    test_df.to_csv(
        os.path.join(output_root, 'test.csv')
        )
    test_df_leaderboard.to_csv(
        os.path.join(output_root, 'test_release_leaderboard.csv')
    )
    joblib.dump(le, os.path.join(output_root, 'le.joblib'))
    

if __name__ == '__main__':
    test_file_path = '/extra/ucinlp0/zhouhanx/react-detection/detection-experiments/hatebase/clean_vs_all/bert'
    main(test_file_path)
    # train_df, test_df = train_test_split_df(
    #     pd.read_csv(test_file_path)
    #     )

    # print(train_df.head(10))
    # print(test_df.head(10))

