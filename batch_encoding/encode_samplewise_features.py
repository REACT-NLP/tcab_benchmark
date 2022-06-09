"""
Do not run this (or run this for testing only)
"""

import pandas as pd
import torch
from tqdm import tqdm
import os
import numpy as np
import hashlib

from reactdetect.feature_extraction import FeatureExtractor
from reactdetect.utils.model_loading import load_target_model
from reactdetect.utils.model_loading import load_uclmr_model
from reactdetect.utils.hashing import hash_pk_tuple, get_pk_tuple
# from reactdetect.allennlp_wrappers.trainer import lazy_groups_of # credit: allennlp

from transformers import AutoModel, AutoTokenizer
from transformers import AutoModel, AutoTokenizer
from transformers import RobertaTokenizer
from transformers import RobertaForMaskedLM
from transformers import GPT2TokenizerFast
from transformers import GPT2LMHeadModel
from transformers import XLNetTokenizerFast
from transformers import XLNetLMHeadModel

assert torch.cuda.is_available(), 'encoding features is quite expensive, defenitely use gpus'
CUDA_DEVICE = torch.device('cuda')


######## MACROS ##############
def no_duplicate_index(df):
    return df.index.is_unique

def get_value_holder(df):
    """
    Given a df, return a dict keyed by index
    """
    out = {}
    for idx in df.index:
        out[idx] = {}
        out[idx]['num_successful_loop'] = 0
        out[idx]['deliverable'] = {}
        PK = get_pk_tuple(df, idx)
        out[idx]['primary_key'] = PK
        out[idx]['unique_id'] = hash_pk_tuple(PK)
    return out

import json
def show_sample_instance(holder, index):
    """
    un-mutating printing util
    """
    out = {}
    out['num_successful_loop'] = holder[index]['num_successful_loop']
    out['primary_key'] = holder[index]['primary_key']
    out['unique_id'] = holder[index]['unique_id']

    out['deliverable'] = {}
    for feat_name in holder[index]['deliverable'].keys():
        feat_shape = 'arr/list of shape: '+str(np.array(holder[index]['deliverable'][feat_name]).shape)
        out['deliverable'][feat_name] = feat_shape
    print(out)




######### HEAVEY LIFTING PORTION FOR ENCODING STUFF #################

def encode_text_properties(df, holder, disable_tqdm=False):
    """
    input is df, return a bool array of len(DF) 

    Text properties should be fairly pain-free,
        aside from BERT that is a bit slow, there shouldnt be too much overhead/loading bunch of external
        models, etc
    """
    print('preparing text properties encoding')
    assert(no_duplicate_index(df))
    
    # we fit vectorizer first, so samples no longer have to be sent in batches
    #   then we can use tqdm, etc for better flexibility
    _ = FeatureExtractor()
    _._fit_vectorizer(df['perturbed_text'])
    
    # define feature extractor
    fe = FeatureExtractor(add_tags=['tp'])

    # commenting out lines 101 and 102 since we're not using tp_ngrams for now

    # fe.vectorizer = _.vectorizer
    # assert fe.vectorizer is not None

    # then load the bert model (out-of-box)
    print('--- loading lm')
    BERT_MODEL_NAME = "sentence-transformers/bert-base-nli-mean-tokens"
    LM_BERT_MODEL = AutoModel.from_pretrained(BERT_MODEL_NAME).to(CUDA_DEVICE)
    LM_BERT_TOKENIZER = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    print('--- lm loaded')
        

    # encode the features
    for idx in tqdm(df.index, disable=disable_tqdm):
        try:
            res = fe(
                return_dict=True,
                text_list=pd.Series([df.at[idx,'perturbed_text']]),
                lm_bert_model=LM_BERT_MODEL,
                lm_bert_tokenizer=LM_BERT_TOKENIZER,
                device=CUDA_DEVICE
                )
            for extractor_name in res.keys():
                dimension_names, values = res[extractor_name][0], res[extractor_name][1]
                holder[idx]['deliverable'][extractor_name] = values
            holder[idx]['num_successful_loop'] += 1
        except Exception as e:
            print('**'*40)
            try:
                print(df.at[idx,'perturbed_text'], ' failed')
            except:
                print("cannot event print offending text somehow, prob. decoding err")
            print('reason: ', e)
            
            

    del LM_BERT_MODEL 
    del LM_BERT_TOKENIZER

    return holder

def encode_lm_perplexity(df, holder,  disable_tqdm=False):
    """
    LM properties envolves loading lm masked model
    """
    print('preparing lm perplexity encoding')
    assert(no_duplicate_index(df))
    
    # define the feature extractor
    fe = FeatureExtractor(add_specific=['lm_perplexity'])

    # then load the language models (out-of-box)
    LM_CAUSAL_MODEL_GPT_NAME = 'gpt2-large'
    LM_CAUSAL_MODEL_GPT = GPT2LMHeadModel.from_pretrained(LM_CAUSAL_MODEL_GPT_NAME).to(CUDA_DEVICE)
    LM_CAUSAL_TOKENIZER_GPT = GPT2TokenizerFast.from_pretrained(LM_CAUSAL_MODEL_GPT_NAME)
    print(LM_CAUSAL_MODEL_GPT_NAME , 'loaded')

    # encode the features
    for idx in tqdm(df.index, disable=disable_tqdm):
        try:
            res = fe(
                return_dict=True,
                text_list=pd.Series([df.at[idx,'perturbed_text']]),
                lm_causal_model=LM_CAUSAL_MODEL_GPT,
                lm_causal_tokenizer=LM_CAUSAL_TOKENIZER_GPT,
                device=CUDA_DEVICE
                )
            for extractor_name in res.keys():
                dimension_names, values = res[extractor_name][0], res[extractor_name][1]
                holder[idx]['deliverable'][extractor_name] = values
            holder[idx]['num_successful_loop'] += 1
        except Exception as e:
            print('**'*40)
            try:
                print(df.at[idx,'perturbed_text'], ' failed')
            except:
                print("cannot event print offending text somehow, prob. decoding err")
            print('reaason: ', e)

    del LM_CAUSAL_MODEL_GPT

    return holder

def encode_lm_proba(df, holder,  disable_tqdm=False):
    """
    LM properties envolves loading lm masked model
    """
    print('preparing lm proba encoding')
    assert(no_duplicate_index(df))
    
    # define the feature extractor
    fe = FeatureExtractor(add_specific=['lm_proba_and_rank'])

    # load mlm
    LM_MASKED_MODEL_ROBERTA_NAME = 'roberta-base'
    LM_MASKED_MODEL_ROBERTA = RobertaForMaskedLM.from_pretrained(LM_MASKED_MODEL_ROBERTA_NAME, return_dict=True).to(CUDA_DEVICE)
    LM_MASKED_TOKENIZER_ROBERTA = RobertaTokenizer.from_pretrained(LM_MASKED_MODEL_ROBERTA_NAME)        
    print(LM_MASKED_MODEL_ROBERTA_NAME, ' loaded ')

    # encode the features
    for idx in tqdm(df.index, disable=disable_tqdm):
        try:
            res = fe(
                return_dict=True,
                text_list=pd.Series([df.at[idx,'perturbed_text']]),
                lm_masked_model=LM_MASKED_MODEL_ROBERTA,
                lm_masked_tokenizer=LM_MASKED_TOKENIZER_ROBERTA,
                device=CUDA_DEVICE
                )
            for extractor_name in res.keys():
                dimension_names, values = res[extractor_name][0], res[extractor_name][1]
                holder[idx]['deliverable'][extractor_name] = values
            holder[idx]['num_successful_loop'] += 1
        except Exception as e:
            print('**'*40)
            try:
                print(df.at[idx,'perturbed_text'], ' failed')
            except:
                print("cannot event print offending text somehow, prob. decoding err")
            print('reason: ', e)
            

    del LM_MASKED_MODEL_ROBERTA

    return holder



def encode_tm_properties(df, holder, react_convention_target_model_folder='./target_models',  disable_tqdm=False):
    
    ################ LOAD TM, DETERMINE MAX_SEQ_LEN AND NUM_LABELS AUTO ####################
    print('preparing tm properties encoding')
    assert(no_duplicate_index(df))
    assert 'target_model_dataset' in df.columns
    assert 'target_model' in df.columns
    assert os.path.exists(react_convention_target_model_folder)

    TARGET_MODEL_NAME = 'roberta' # HARD CODING FOR NOW, ASSUME ROBERTA IS ATTACKED IN ALL SITUATION
    _ = set(df['target_model_dataset'])
    assert( len(set(df['target_model_dataset'])) ) == 1
    TARGET_MODEL_DATASET = list(_)[0]
    if TARGET_MODEL_DATASET == 'fnc1':
        TARGET_MODEL_NAME = 'uclmr'
    print('--- your tm to extract feature is ', TARGET_MODEL_NAME, ' trained on ', TARGET_MODEL_DATASET)

    
    NUM_LABELS_LOOKUP = {
    'fnc1':4,
    'civil_comments':2,
    'hatebase':2,
    'wikipedia':2,
    'sst':2,
    'imdb':2,
    'climate-change_waterloo':3,
    'nuclear_energy':3,
    'gab_dataset': 2,
    'reddit_dataset': 2,
    'wikipedia_personal': 2
    }
    # lookup how many labels are there
    NUM_LABELS = NUM_LABELS_LOOKUP[TARGET_MODEL_DATASET]

    # prepare params to load that model
    if TARGET_MODEL_DATASET in ['wikipedia_personal', 'reddit_dataset', 'gab_dataset']:
        TARGET_MODEL_DIR = os.path.join(react_convention_target_model_folder, 'hatebase', TARGET_MODEL_NAME)
        assert os.path.exists(TARGET_MODEL_DIR), 'this is not a valid tm directory: '+TARGET_MODEL_DIR
        TARGET_MODEL_METADATA_DIR = os.path.join(react_convention_target_model_folder, 'hatebase', TARGET_MODEL_NAME, 'results.npy')
        assert os.path.exists(TARGET_MODEL_METADATA_DIR), 'dir <'+TARGET_MODEL_METADATA_DIR+"> do not exist!"
    else:
        TARGET_MODEL_DIR = os.path.join(react_convention_target_model_folder, TARGET_MODEL_DATASET, TARGET_MODEL_NAME)
        assert os.path.exists(TARGET_MODEL_DIR), 'this is not a valid tm directory: '+TARGET_MODEL_DIR
        TARGET_MODEL_METADATA_DIR = os.path.join(react_convention_target_model_folder, TARGET_MODEL_DATASET, TARGET_MODEL_NAME, 'results.npy')
        assert os.path.exists(TARGET_MODEL_METADATA_DIR), 'dir <'+TARGET_MODEL_METADATA_DIR+"> do not exist!"

    _ = np.load(TARGET_MODEL_METADATA_DIR, allow_pickle=True)
    MAX_SEQ_LEN = _.item().get('max_seq_len')
    assert( type(MAX_SEQ_LEN) is int), 'oops. somhow max_seq_len in your resutls.npy is not a int'

    if TARGET_MODEL_NAME != 'uclmr':
        TARGET_MODEL = load_target_model(target_model_name=TARGET_MODEL_NAME, dir_target_model=TARGET_MODEL_DIR, num_labels=NUM_LABELS, max_seq_len=MAX_SEQ_LEN, device=CUDA_DEVICE)
        regions = [(0.0, 0.25), (0.25, 0.75), (0.75, 1.0), (0.0, 1.0)]
    else:
        TARGET_MODEL_DIR = os.path.join(react_convention_target_model_folder, TARGET_MODEL_DATASET, 'uclmr')
        TARGET_MODEL = load_uclmr_model(target_model_name='uclmr',
                                        dir_target_model=TARGET_MODEL_DIR,
                                        device=CUDA_DEVICE)
        regions = [(0.0, 1.0)]

    print(' --- target model loaded')

    ######################################################################


    # define the feature extractor
    fe = FeatureExtractor(add_tags=['tm'])
    assert no_duplicate_index(df)
    
    # encode the features
    for idx in tqdm(df.index, disable=disable_tqdm):
        try:
            perturbed_text = df.at[idx,'perturbed_text'] 
            perturbed_output = np.argmax(df.at[idx,'perturbed_output']) 
            res = fe(
                return_dict=True,
                text_list=pd.Series([perturbed_text]),
                labels=pd.Series([perturbed_output]),
                target_model=TARGET_MODEL,
                device=CUDA_DEVICE,
                regions=regions
                )
            for extractor_name in res.keys():
                dimension_names, values = res[extractor_name][0], res[extractor_name][1]
                holder[idx]['deliverable'][extractor_name] = values
            holder[idx]['num_successful_loop'] += 1
        except Exception as e:
            print('**'*40)
            try:
                print(df.at[idx,'perturbed_text'], ' failed')
            except:
                print("cannot event print offending text somehow, prob. decoding err")
            print('reason: ', e)

    return holder
            


def encode_all_properties(df, disable_tqdm=False):
    """
    Takes in a df,
    returns a nested dict called hodler, as the data object
    """
    assert(no_duplicate_index(df))
    HOLDER = get_value_holder(df)
    
    HOLDER = encode_text_properties(df, HOLDER, disable_tqdm=False)
    HOLDER = encode_lm_perplexity(df, HOLDER, disable_tqdm=False)
    HOLDER = encode_lm_proba(df, HOLDER, disable_tqdm=False)
    HODLER = encode_tm_properties(df, HOLDER, disable_tqdm=False)
    LOOP_NUM = 4 #4 extactor pipes

    print('='*40)
    print('--- all done')
    failed_extraction_count = 0
    
    KEYS_TO_RM = []
    for h in HOLDER.keys():
        if HOLDER[h]['num_successful_loop'] == LOOP_NUM:
            pass 
        else:
            KEYS_TO_RM.append(h)
    for k in KEYS_TO_RM:
        del HOLDER[k]
        failed_extraction_count += 1        
    
    print('total failed extraction: ', failed_extraction_count, 'out of', len(HOLDER))
    print('a sample holder value for sanity check')
    print()
    print()
    sample_holder_item_key = list(HOLDER.keys())[0]
    show_sample_instance(HOLDER, sample_holder_item_key)
    print()

    print('='*40)

    return HOLDER

if __name__ == '__main__':

    ifile_path = './test_dataset.csv'
    df = pd.read_csv(ifile_path).head(10)
    print('num instance: ', len(df))
    df = encode_all_properties(df)
    print(df.head())
