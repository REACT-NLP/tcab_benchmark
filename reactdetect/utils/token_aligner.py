import nltk
nltk.download('punkt')

import numpy as np

from nltk import align
from nltk import word_tokenize
from transformers import BertTokenizer
from transformers import RobertaTokenizer
from transformers import XLNetTokenizer
import sys

def convert_to_nltk(raw_sentence, token_vectors, target_model='bert'):

    # make sure target models are either of these
    assert target_model=='bert' or target_model=='roberta' or target_model=='xlnet'

    if target_model=='bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif target_model=='roberta':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    else:
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

    
    nltk_tokens = word_tokenize(raw_sentence) # define nltk tokens for the raw sentence
    tm_tokens = tokenizer.tokenize(raw_sentence) # define target model tokens for the raw sentence
    # print(nltk_tokens, len(nltk_tokens))
    # print(tm_tokens, len(tm_tokens))
    # sys.exit()

    try:
        assert len(tm_tokens) == len(token_vectors) # number of target model tokens should be equal to length of token_vectors
    except:
        print(tm_tokens)
        print(token_vectors.shape)
        raise
    
    nltk_to_tm_tokens, _ = align(nltk_tokens, tm_tokens) # find an index mapping from nltk tokens to target model tokens
    # print(nltk_to_tm_tokens)
    # sys.exit()

    new_token_vectors = []

    i = 0
    while i<len(nltk_tokens):
        print(nltk_to_tm_tokens)
        l = []
        for j in nltk_to_tm_tokens[i]:
            l.append(token_vectors[j])
        if len(l)==0:
            nltk_to_tm_tokens[i].append(j+1)
            continue
        if len(l)==1:
            new_token_vectors.append(l[0])
        else:
            if isinstance(l[0], list):
                new_token_vectors.append(list(np.average(l, axis=0)))
            else:
                new_token_vectors.append(np.average(l, axis=0))
        i+=1


    return new_token_vectors

if __name__=='__main__':
    test_sentence = """'Dieses ‟ Afghan scenario roler implies be weak Statwith nominate power about effectively autonomous Powerversigarea wehrgeslead ed strongald who et verfügen in Concentral government .'"""
    token_vectors = [[0.1, 0.2] for i in range(43)]
    print(convert_to_nltk(test_sentence, token_vectors))

        