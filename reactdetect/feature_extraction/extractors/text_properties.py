import numpy as np
import pandas as pd
import torch
from math import ceil
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler

from .utils import *

LOWERCASE_LETTERS = 'abcdefghijklmnopqrstuvwxyz'

np.seterr(invalid='raise')


@apply_tags(['tp', 'output_capable'])
def tp_num_chars(text_list, feature_list=None):
    """
    Input: Pandas.Series of strings.
    Returns: 2D Numpy.Array of shape=(no. samples, 1);
             character counts.
    """
    if type(feature_list) == list:
        feature_list.append('num_chars')

    assert type(text_list) == pd.Series
    return text_list.apply(lambda s: len(s)).values.reshape(-1, 1)


@apply_tags(['tp', 'output_capable'])
def tp_num_alpha_chars(text_list, feature_list=None):
    """
    Input: Pandas.Series of strings.
    Returns: 2D Numpy.Array of shape=(no. samples, 1);
             no. alpha characters.
    """
    if type(feature_list) == list:
        feature_list.append('num_alpha_chars')

    assert type(text_list) == pd.Series
    return text_list.apply(lambda s: sum(c.isalpha() for c in s)).values.reshape(-1, 1)


@apply_tags(['tp', 'output_capable'])
def tp_num_digits(text_list, feature_list=None):
    """
    Input: Pandas.Series of strings.
    Returns: 2D Numpy.Array of shape=(no. samples, 1);
             no. digits.
    """
    if type(feature_list) == list:
        feature_list.append('num_digits')

    assert type(text_list) == pd.Series
    return text_list.apply(lambda s: sum(c.isdigit() for c in s)).values.reshape(-1, 1)


@apply_tags(['tp', 'output_capable'])
def tp_num_punctuation(text_list, feature_list=None):
    """
    Input: Pandas.Series of strings.
    Returns: 2D Numpy.Array of shape=(no. samples, 1);
             no. punctuation marks.
    """
    if type(feature_list) == list:
        feature_list.append('num_punctuation')

    assert type(text_list) == pd.Series
    count = lambda l1, l2: sum([1 for x in l1 if x in l2])
    return text_list.apply(lambda s: count(s, set(string.punctuation))).values.reshape(-1, 1)


@apply_tags(['tp', 'output_capable'])
def tp_num_multi_spaces(text_list, feature_list=None):
    """
    Input: Pandas.Series of strings.
    Returns: 2D Numpy.Array of shape=(no. samples, 1);
             no. punctuation marks.
    """
    if type(feature_list) == list:
        feature_list.append('num_multi_spaces')

    assert type(text_list) == pd.Series

    # define multi spaces
    def count_multi_spaces(s):
        result = 0
        activator = 0
        prev_isspace = s[0].isspace()
        for c in s[1:]:
            if c.isspace():
                if prev_isspace and activator == 0:
                    result += 1
                    activator = 1
            else:
                activator = 0
            prev_isspace = c.isspace()
        return result

    return text_list.apply(lambda s: count_multi_spaces(s)).values.reshape(-1, 1)


@apply_tags(['tp', 'output_capable'])
def tp_num_words(text_list, feature_list=None):
    """
    Input: Pandas.Series of strings.
    Returns: 2D Numpy.Array of shape=(no. samples, 1);
             word counts.
    """
    if type(feature_list) == list:
        feature_list.append('num_words')

    assert type(text_list) == pd.Series
    return text_list.apply(lambda s: len(s.split())).values.reshape(-1, 1)


@apply_tags(['tp', 'output_capable'])
def tp_avg_word_length(text_list, quantiles=[0.25, 0.5, 0.75],
                    regions=[(0, 0.25), (0.25, 0.75), (0.75, 1.0), (0.0, 1.0)],
                    feature_list=None):
    """
    Input: Pandas.Series of strings.
    Returns: 2D Numpy.Array of shape=(no. samples, (1 + 1 + no. quantiles) * no. regions)
             mean, variance, and quantiles of the word length vector
             for different input regions.
    """
    if type(feature_list) == list:
        for i in range(len(regions)):
            feature_list.append('avg_word_length_mean_region{}'.format(i))
            feature_list.append('avg_word_length_var_region{}'.format(i))
            feature_list += ['avg_word_length_quant{}_region{}'.format(j, i) for j in range(len(quantiles))]

    assert type(text_list) == pd.Series
    feature_vec_list = []

    # compute word length statistics for each sample
    for text_str in text_list.tolist():
        feature_vec = []

        # compute length of each word
        word_lengths = [len(w) for w in remove_punctuation(text_str).split()]

        # compute statistics for different input regions
        for start_frac, end_frac in regions:

            # get region start and end indices
            start_ndx = int(start_frac * len(word_lengths))
            end_ndx = int(ceil(end_frac * len(word_lengths))) #round up to avoid empty region

            # compute statistics for this region
            region_word_lengths = word_lengths[start_ndx:end_ndx]
            feature_vec.append(np.mean(region_word_lengths))
            feature_vec.append(np.var(region_word_lengths))
            try:
                feature_vec += list(np.quantile(region_word_lengths, quantiles))
            except IndexError:
                feature_vec += [0] * len(quantiles)

        # add feature vector
        feature_vec_list.append(feature_vec)

    return np.vstack(feature_vec_list)


@apply_tags(['tp', 'output_capable'])
def tp_num_non_ascii(text_list, feature_list=None):
    """
    Input: Pandas.Series of strings.
    Returns: 2D Numpy.Array of shape=(no. samples, 1);
             non-ascii char. counts.
    """
    if type(feature_list) == list:
        feature_list.append('num_non_ascii')

    assert type(text_list) == pd.Series
    ascii_chars = [chr(i) for i in range(127)]
    non_ascii_count = lambda s: sum(c not in ascii_chars for c in s.replace(' ', ''))
    return text_list.apply(non_ascii_count).values.reshape(-1, 1)


@apply_tags(['tp', 'output_capable'])
def tp_num_cased_letters(text_list, feature_list=None):
    """
    Input: Pandas.Series of strings.
    Returns: 2D Numpy.Array of shape=(no. samples, 4);
             counts of uppercase letters,
             counts of lowercase letters,
             fracion of uppercase letters,
             fraction of lowercase letters.
    """
    if type(feature_list) == list:
        feature_list += ['num_uppercase_letters', 'num_lowercase_letters',
                         'fraction_uppercase_letters', 'fraction_lowercase_letters']

    assert type(text_list) == pd.Series

    # define counts
    upper_count = lambda s: sum(1 if c.isupper() else 0 for c in s)
    lower_count = lambda s: sum(1 if c.islower() else 0 for c in s)

    # compute counts
    upper_count_vals = text_list.apply(upper_count).values
    lower_count_vals = text_list.apply(lower_count).values

    total_count_vals = upper_count_vals + lower_count_vals

    upper_count_frac = np.divide(upper_count_vals, total_count_vals, where=upper_count_vals != 0)
    lower_count_frac = np.divide(lower_count_vals, total_count_vals, where=lower_count_vals != 0)

    # organize features
    features = []
    features.append(upper_count_vals.reshape(-1, 1))
    features.append(lower_count_vals.reshape(-1, 1))
    features.append(upper_count_frac.reshape(-1, 1))
    features.append(lower_count_frac.reshape(-1, 1))

    return np.hstack(features)


@apply_tags(['tp', 'output_capable'])
def tp_is_first_word_lowercase(text_list, feature_list=None):
    """
    Input: Pandas.Series of strings.
    Returns: 2D Numpy.Array of shape=(no. samples, 1);
             1 if lowercase first letter in for the first word.
    """
    if type(feature_list) == list:
        feature_list.append('is_first_word_lowercase')

    assert type(text_list) == pd.Series

    def first_lower(s):
        try:
            lower = remove_punctuation(s).split()[0][0].islower()
            return lower
        except IndexError:
            return 0

    return text_list.apply(first_lower).values.reshape(-1, 1)


@apply_tags(['tp', 'output_capable'])
def tp_num_mixed_case_words(text_list, feature_list=None):
    """
    Input: Pandas.Series of strings.
    Returns: 2D Numpy.Array of shape=(no. samples, 1);
             no. words with upper and lowercase letters.
    """
    if type(feature_list) == list:
        feature_list.append('num_mixed_case_words')

    assert type(text_list) == pd.Series

    # define a mixed-case word
    def is_mixed_case(w):
        isupper = 0
        islower = 1 if w[0].islower() else 0
        for c in w[1:]:  # disregard first letter
            if c.islower():
                islower = 1
            elif c.isupper():
                isupper = 1
            if islower and isupper:
                return 1
        return 0

    mixed_case = lambda s: sum(is_mixed_case(w) for w in remove_punctuation(s).split() if len(w) > 1)
    return text_list.apply(mixed_case).values.reshape(-1, 1)


@apply_tags(['tp', 'output_capable'])
def tp_num_single_lowercase_letters(text_list, feature_list=None):
    """
    Input: Pandas.Series of strings.
    Returns: 2D Numpy.Array of shape=(no. samples, 1);
             no. single lowercase letters that are not 'a' or 'i'.
    """
    if type(feature_list) == list:
        feature_list.append('num_single_lowercase_letters')

    assert type(text_list) == pd.Series
    is_lower = lambda c: 1 if c in LOWERCASE_LETTERS and c not in 'ai' else 0
    single_lower = lambda s: sum(is_lower(w) for w in remove_punctuation(s).split() if len(w) == 1)
    return text_list.apply(single_lower).values.reshape(-1, 1)


@apply_tags(['tp', 'output_capable'])
def tp_num_lowercase_after_punctuation(text_list, feature_list=None):
    """
    Input: Pandas.Series of strings.
    Returns: 2D Numpy.Array of shape=(no. samples, 1);
             counts of lowercase letters after a punctuation char.
    """
    if type(feature_list) == list:
        feature_list.append('num_lowercase_letters_after_punctuation')

    assert type(text_list) == pd.Series
    punctuation = '.!?'

    # compute feature for each sample
    value_list = []
    for text in text_list.tolist():

        # split text into words
        word_list = text.split()
        prev_has_punctuation = word_list[0][-1] in punctuation

        # count words that start lowercase after punctuation
        cnt = 0
        for word in word_list[1:]:
            if prev_has_punctuation and word[0].islower():
                cnt += 1
            prev_has_punctuation = word[-1]in punctuation

        # add feature to list of features
        value_list.append(cnt)

    return np.array(value_list).reshape(-1, 1)


@apply_tags(['tp'])
def tp_num_cased_word_switches(text_list, feature_list=None):
    """
    Input: Pandas.Series of strings.
    Returns: 2D Numpy.Array of shape=(no. samples, 1);
             no. times words switch from being all uppercase
               to all lowercase or vice versa.
    """
    if type(feature_list) == list:
        feature_list.append('num_cased_word_switches')

    assert type(text_list) == pd.Series

    value_list = []
    for text in text_list.tolist():
        word_list = remove_numbers(remove_punctuation(text)).split()

        cnt = 0
        if len(word_list) == 0:
            value_list.append(cnt)
            break

        prev_word = word_list[0]

        # count no. times casing switches
        for word in word_list:
            if prev_word.isupper() and word.islower():
                cnt += 1
            elif prev_word.islower() and word.isupper():
                cnt += 1
            prev_word = word

        value_list.append(cnt)

    return np.array(value_list).reshape(-1, 1)


# @apply_tags(['tp', 'output_capable'])
# def tp_ngrams(text_list, analyzer='char_wb', ngram_range=(3, 3), max_features=5000,
#            lowercase=False, use_idf=False, vectorizer=None, logger=None, return_vectorizer=False,
#            feature_list=None):
#     """
#     Input: Pandas.Series of strings.
#         If return_vectorizer=True, this function will return the vectorizer AND the ngram features,
#         otherwise, only the result will be returned — this is so that the function has the ability to work
#         with the FeatureExtractor.__call__ method.
#     Returns: 2D Numpy.Array of shape=(len(text_list), max_features);
#              ngrams,
#              trained vectorizer.
#     """

#     if vectorizer is None:
#         vectorizer = TfidfVectorizer(stop_words='english',
#                                      ngram_range=ngram_range,
#                                      analyzer=analyzer,
#                                      max_features=max_features,
#                                      lowercase=lowercase,
#                                      use_idf=use_idf)

#         result = vectorizer.fit_transform(text_list).toarray()

#     else:
#         result = vectorizer.transform(text_list).toarray()

#     # only add as many features as there are columns in the vectorized output
#     if type(feature_list) == list:
#         feature_list += ['ngrams{}'.format(i) for i in range(result.shape[1])]

#     # to make the function work as brophy initially designed it, set the return_vectorizer argument to True
#     if return_vectorizer:
#         return result, vectorizer
#     else:
#         return result


@apply_tags(['tp'])
def tp_bert(text_list, lm_bert_model, lm_bert_tokenizer, device, max_length=128, batch_size=50, feature_list=None):
    """
    Input:
        :param text_list: Pandas.Series of strings;
        :param lm_bert_model: BERT model (e.g. model returned by:
                AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens"))
        :param lm_bert_tokenizer: Transformers tokenizer (e.g. tokenizer returned by:
                AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
        :param device: torch.device object
        :param max_length: The maximum input sequence length for the model.
    Returns: 2D Numpy.Array of shape=(no. samples, m));
             perplexity of the input sequence for different input regions.
    Reference: https://github.com/zhouhanxie/react-detection/blob/main/lineardetect-bert.py, or
        https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens
    """
    if type(feature_list) == list:
        # feature_list.append('lm_bert')
        for i in range(768):
            feature_list.append('lm_bert_{}'.format(i))

    # prepare data
    data = text_list.tolist()
    dataloader = DataLoader(data, sampler=SequentialSampler(data), batch_size=batch_size)

    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    # Iterate through data
    outputs = []
    for i, batch_text in enumerate(dataloader):

        # Tokenize sentences in this batch and send to device
        encoded_input = lm_bert_tokenizer(batch_text, padding=True, truncation=True,
                                          max_length=max_length, return_tensors='pt')
        encoded_input = encoded_input.to(device)

        # Compute token embeddings for this batch
        with torch.no_grad():
            model_output = lm_bert_model(**encoded_input)

        # Get sentence embeddings for this batch
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask']).cpu().numpy()

        outputs.append(sentence_embeddings)

    # Aggregate outputs into single numpy array
    outputs = np.vstack(outputs)

    return outputs