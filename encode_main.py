import pandas as pd
import tqdm
from collections import Counter
from reactdetect.utils.pandas_ops import restrict_max_instance_for_class
from reactdetect.utils.magic_vars import SUPPORTED_TARGET_MODELS, SUPPORTED_ATTACKS, SUPPORTED_TARGET_MODEL_DATASETS
import os

def mkfile_if_dne(fpath):
    if not os.path.exists(os.path.dirname(fpath)):
        print('mkfile warning: making fdir since DNE: ',fpath)
        os.makedirs(os.path.dirname(fpath))
    else:
        pass

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

def address_clean_samples(df):
    for idx in df.index:
        if df.at[idx,'status'] == 'clean':
            df.at[idx,'perturbed_text'] = df.at[idx,'original_text']
            df.at[idx,'attack_name'] = 'clean'
    return df

def holder_to_disk(holder,  fname):
    """
    holder is a nested dict, see encode_samplewise_features.py

    pushes items in this to mongodb
    """
    import joblib
    joblib.dump(holder, fname)



# from MONGO_USER_CONFIG import uri,username,password,db_name,sample_wise_collection_name
# from MongoWrapper import MongoWrapper 
# def holder_to_mongo(holder, disable_tqdm=False):
#     """
#     holder is a nested dict, see encode_samplewise_features.py

#     pushes items in this to mongodb
#     """
#     print('--setting up connection')
    
#     db = MongoWrapper(db_name=db_name, collection_name=sample_wise_collection_name, hostname=uri, port=0,
#                   username=username, password=password) 
    
    
#     R = []
#     print('\nadding instances...')
#     from tqdm import tqdm
#     for k in  tqdm(list(holder.keys()), disable=disable_tqdm):
#         instance_dict = holder[k]['deliverable']
#         instance_dict['_id'] = holder[k]['unique_id']

#         PK_MEANINGS = sorted(['scenario','target_model_dataset', 'target_model','attack_toolchain','attack_name', 'test_index'])
#         for i in range( len(holder[k]['primary_key'])) :
#             instance_dict[PK_MEANINGS[i]] = holder[k]['primary_key'][i]
        
        
#         r = db.save(instance_dict)  # save instance into the database
#         R.append(r)

#     print('closing connection')
#     db._close() 
#     print('finished')


if __name__ == '__main__':

    # arg sanity checks

    import argparse

    # must have cms
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_model', type=str, default=None, help='target model name, must be the same as dir in dropbox')
    parser.add_argument('--target_model_dataset', type=str, default=None, help='target model dataset name, must be the same as dir in dropbox')
    parser.add_argument('--target_model_train_dataset', type=str, default=None, help='dataset used to train target model, must be the same as dir in dropbox')
    parser.add_argument('--attack_name', type=str, default=None, help="""attack name, use string  clean for clean """)
    parser.add_argument('--to_mongo', type=str, default='No', help='if True saves to mongo, else save to disk')
    parser.add_argument('--max_clean_instance', type=int, default=999999, help='only consider certain number of clean instances')
    
    # other might be useful cmds
    parser.add_argument('--test', type=bool, default=False, help='if True only computes first 10 instance')
    parser.add_argument('--disable_tqdm', type=bool, default=False, help='if True silent tqdm progress bar')
    parser.add_argument('--uo_cache', action='store_true',
                        help='if provided, change the cache to a location that has more space for UO users')
    

    # get args, sanity check
    args = parser.parse_args()
    if args.uo_cache:
        cache_dir = '.cache'
        os.environ['TRANSFORMERS_CACHE'] = cache_dir  # assumes UO students are working in /projects/uoml/<username>
        print(f'changed transformers cache directory to: {cache_dir}')
    assert args.target_model in SUPPORTED_TARGET_MODELS 
    assert args.target_model_dataset in SUPPORTED_TARGET_MODEL_DATASETS
    assert args.target_model_train_dataset in SUPPORTED_TARGET_MODEL_DATASETS
    assert args.attack_name in SUPPORTED_ATTACKS or args.attack_name == 'ALL' or args.attack_name == 'ALLBUTCLEAN'
    assert type(args.max_clean_instance) == int
    print('push to mongo? ', args.to_mongo)


    
    # io
    print('--- reading csv')
    DF=pd.read_csv('whole_catted_dataset.csv')
    print()
    print('--- stats before filtering')
    print(show_df_stats(DF))
    print('--- filtering dataframe')
    
    # compute
    # DF = address_clean_samples(DF) # make clean instance in same format, no longer needed
    if args.attack_name == 'ALL':
        print('--- attack name is ALL, using all attacks')
        DF = DF[(DF['target_model_dataset'] == args.target_model_dataset) & (DF['target_model_train_dataset'] == args.target_model_train_dataset) & (DF['target_model'] == args.target_model)]
    elif args.attack_name == 'ALLBUTCLEAN':
        print('--- attack name is ALLBUTCLEAN, using all attacks but clean')
        DF = DF[(DF['target_model_dataset'] == args.target_model_dataset) & (DF['target_model_train_dataset'] == args.target_model_train_dataset) & (DF['target_model'] == args.target_model) & (DF['attack_name'] != 'clean')]
    else:
        DF = DF[(DF['target_model_dataset'] == args.target_model_dataset) &  (DF['target_model_dataset'] == args.target_model_dataset) & (DF['target_model'] == args.target_model) & (DF['attack_name'] == args.attack_name)]

    print(' done , instance distribution: ')
    print(show_df_stats(DF))

    print('--- (potentially) dropping clean instance to ', args.max_clean_instance)
    print(' done , instance distribution: ')
    DF = restrict_max_instance_for_class(in_df=DF,attack_name_to_clip='clean', max_instance_per_class=args.max_clean_instance)
    print(show_df_stats(DF))

    print()
    print('--- starting the encoding process')
    from batch_encoding.encode_samplewise_features import encode_all_properties
    
    # if test use only 10 sample
    if args.test:
        print('*** WARNING, TEST MODE, only encode 10 samples')
        DF = DF.head(10)
    
    # encode everything. DF in, dict out
    HOLDER = encode_all_properties(DF, disable_tqdm=args.disable_tqdm)


    print('-- (always) saving to disk')
        
    if args.test:
        fname = '_'.join( ['test',args.target_model,args.target_model_dataset,args.attack_name,'.joblib']).strip()
    else:
        fname = '_'.join( [args.target_model,args.target_model_dataset,args.attack_name,'.joblib']).strip()
    fname = os.path.join('./reprs/samplewise', fname)
    mkfile_if_dne(fname)
    holder_to_disk(HOLDER, fname)
    
    
    # # save
    # if args.to_mongo == 'Yes':
    #     print('-- pushing to mongo')
    #     holder_to_mongo(HOLDER, disable_tqdm=args.disable_tqdm)
    # else:
    #     print(' to_mongo is no, exit.')
    
        
            

    

    