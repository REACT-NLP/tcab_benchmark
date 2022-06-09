"""
Note: Don't have to run this if you have
    pre-generated whole catted dataset already under ./reactdetect
"""

########## GLOBAL VARS ########

RANDOM_SEED = 1 # make data sampling replaicatable 


########## MACROS #############

import os

def mkfile_if_dne(fpath):
    if not os.path.exists(os.path.dirname(fpath)):
        print('mkfile warning: making fdir since DNE: ',fpath)
        os.makedirs(os.path.dirname(fpath))
    else:
        pass

def grab_csvs(root_dir):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(root_dir):
        for file in f:
            if 'csv' in file:
                files.append(os.path.join(r, file))
    return files

import  sys
def path_splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

import pandas as pd
import yaml
def to_data_info(csv_path):
    out = {}
    allparts = path_splitall(csv_path)
    out['data'] = pd.read_csv(csv_path)
    return out

def concat_multiple_df(list_of_dfs):
    return refresh_index(pd.concat(list_of_dfs, axis=0))

def refresh_index(df):
    return df.reset_index(drop=True)


from collections import Counter
def no_duplicate_index(df):
    return df.index.is_unique

def no_duplicate_preturbed_text(df, column_name):
    count = Counter(df[column_name])
    for sent in count.keys():
        if count[sent]>1:
            return False
    return True

def drop_for_column_outside_of_values(df, column_name, values):
    return df[df[column_name].isin(values)]

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

# input - df: a Dataframe, chunkSize: the chunk size
# output - a list of DataFrame
# purpose - splits the DataFrame into smaller chunks
# credit - https://stackoverflow.com/questions/17315737/split-a-large-pandas-dataframe
def split_dataframe(df, chunk_size = 10000): 
    chunks = list()
    num_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size else 0)
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

def all_cols_nan_to_strnone(df):
    columns = df.columns 
    instruction = {}
    for cn in columns:
        instruction[cn] = 'None' 
    return df.fillna(instruction)

def address_clean_samples(df):
    for idx in df.index:
        if df.at[idx,'status'] == 'clean':
            df.at[idx,'perturbed_text'] = df.at[idx,'original_text']
            df.at[idx,'attack_name'] = 'clean'
    return df


if __name__ == '__main__':


    good_to_go = input('****** warning, do not run this if you are using downloaded whole_catted_dataset.csv form drive, still run (Y/N)?')
    if good_to_go != 'Y':
        raise KeyboardInterrupt

    print('--- loading all data')
    csvs = grab_csvs('./attacks_dataset/attacks')
    datas = []
    for csv in csvs:
        info = to_data_info(csv)
        dat = info['data'] 
        datas.append(dat)
    whole_catted_dataset = concat_multiple_df(datas)
    assert('test_ndx' not in whole_catted_dataset.keys() )
    assert('dataset' not in whole_catted_dataset.keys() )
    whole_catted_dataset['test_ndx'] = whole_catted_dataset['test_index']
    whole_catted_dataset['dataset'] = whole_catted_dataset['target_model_dataset']
    whole_catted_dataset = refresh_index(whole_catted_dataset)
    whole_catted_dataset = all_cols_nan_to_strnone(whole_catted_dataset)
    whole_catted_dataset = address_clean_samples(whole_catted_dataset)
    print('done, all data statistics: ')
    print(show_df_stats(whole_catted_dataset))

    # whole_catted_dataset.to_csv('whole_catted_dataset_unfiltered.csv')
    
    print('--- doing global filtering over all data')
    print('dropping unsuccessful attacks...')
    whole_catted_dataset = drop_for_column_outside_of_values(whole_catted_dataset, 'status', ['success', 'clean'])

    print('dropping invalid attacks...')
    VALID_ATTACKS = ['bae','deepwordbug', 'faster_genetic', 'genetic', 'hotflip', 'iga_wang', 'pruthi', 'pso', 'textbugger', 'textfooler', 'viper','clean', 'deepwordbugv1', 'deepwordbugv2', 'deepwordbugv3', 'pruthiv1', 'pruthiv2', 'pruthiv3', 'textbuggerv1', 'textbuggerv2', 'textbuggerv3']
    whole_catted_dataset = drop_for_column_outside_of_values(whole_catted_dataset, 'attack_name', VALID_ATTACKS)
    
    whole_catted_dataset = refresh_index(whole_catted_dataset)
    print('done, all data statistics: ')
    print(show_df_stats(whole_catted_dataset))

    print('-- saving to disk')
    whole_catted_dataset.to_csv('whole_catted_dataset.csv')

    # print('-- saving a smaller test data to disk')
    # whole_catted_dataset.head(100).to_csv('test_dataset.csv')