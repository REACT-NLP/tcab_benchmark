import torch
import joblib
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from reactdetect.utils.file_io import grab_fpath_with
from reactdetect.feature_extraction import FeatureExtractor
from reactdetect.utils.model_loading import load_target_model
from transformers import AutoModel, AutoTokenizer

def load_bert_detector(bert_detection_model_root):

    if 'clean_vs_all' in bert_detection_model_root:
        num_labels = 2
    else:
        num_labels = len(pd.read_csv(
            os.path.join(bert_detection_model_root, 'train.csv')
        )['attack_name'].unique())
        
    model_metadata = np.load(
        os.path.join(bert_detection_model_root, 'results.npy'),
        allow_pickle = True
    ).item()
    max_seq_len = model_metadata['max_seq_len']
    target_model_name = 'bert'
    dir_target_model = bert_detection_model_root


    bert_detector = load_target_model(
        target_model_name=target_model_name, 
        dir_target_model=dir_target_model,  
        max_seq_len=max_seq_len,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
        num_labels=num_labels
    )
    
    return bert_detector

def re_encode_with_bertclassifier(data_dict, bert_detector):
    with torch.no_grad():
        for key in tqdm(list(data_dict.keys())):
            print(key, 'key in data_dict')
            instance = data_dict[key]
            old_bert_shape = instance['deliverable']['tp_bert'].shape
            new_tp_bert = bert_detector.get_cls_repr([instance['perturbed_text']]).cpu().numpy()
            assert new_tp_bert.shape == old_bert_shape
            instance['deliverable']['tp_bert'] = new_tp_bert 
    return data_dict