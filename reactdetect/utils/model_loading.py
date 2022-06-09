# target models, for tm properties
import pickle
from transformers import RobertaTokenizer
from transformers import RobertaForMaskedLM
from reactdetect.models.target_models import RoBERTaClassifier, XLNetClassifier, UCLMRClassifier, BERTClassifier
import torch
import os


def get_model(model_name, max_seq_len=250, num_labels=2, tf_vectorizer=None, tfidf_vectorizer=None):
    """
    Return a new instance of the text classification model.
    """
    if model_name == 'bert':
        model = BERTClassifier(max_seq_len=max_seq_len, num_labels=num_labels)

    elif model_name == 'roberta':
        model = RoBERTaClassifier(max_seq_len=max_seq_len, num_labels=num_labels)

    elif model_name == 'xlnet':
        model = XLNetClassifier(max_seq_len=max_seq_len, num_labels=num_labels)

    elif model_name == 'uclmr':
        assert tf_vectorizer is not None
        assert tfidf_vectorizer is not None
        model = UCLMRClassifier(tf_vectorizer=tf_vectorizer, tfidf_vectorizer=tfidf_vectorizer)

    else:
        raise ValueError('Unknown model {}!'.format(model_name))

    return model

def load_target_model(target_model_name, dir_target_model,  max_seq_len , device, num_labels=2):
    """
    Loads trained model weights and sets model to evaluation mode.
    """

    # load trained model
    model = get_model(model_name=target_model_name, num_labels=num_labels,max_seq_len=max_seq_len)
    model.load_state_dict(torch.load(os.path.join(dir_target_model, 'weights.pt'), map_location=device))
    model = model.to(device)
    model.eval()

    return model

def load_uclmr_model(target_model_name, dir_target_model, device='cpu'):
    """
    Loads trained model weights and sets model to evaluation mode.
    """

    # load vectorizers
    tf_vectorizer = pickle.load(open(os.path.join(dir_target_model, 'tf_vectorizer.pkl'), 'rb'))
    tfidf_vectorizer = pickle.load(open(os.path.join(dir_target_model, 'tfidf_vectorizer.pkl'), 'rb'))

    # load trained model
    model = get_model(model_name=target_model_name, tf_vectorizer=tf_vectorizer, tfidf_vectorizer=tfidf_vectorizer)
    model.load_state_dict(torch.load(os.path.join(dir_target_model, 'weights.pt'), map_location=device))
    model = model.to(device)
    model.eval()

    return model

def find_held_tm(target_models):
    if target_models in ('bert+roberta','roberta+bert'):
        held_tm = 'xlnet'
    elif target_models in ('bert+xlnet', 'xlnet+bert'):
        held_tm = 'roberta'
    elif target_models in ('roberta+xlnet', 'xlnet+roberta'):
        held_tm = 'bert'
    return held_tm
