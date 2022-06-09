import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler
import bert_score

from .utils import apply_tags, split_sentences


@apply_tags(['lm'])
def lm_proba_and_rank(text_list, lm_masked_model, lm_masked_tokenizer, device,
                      logger=None, quantiles=[0.25, 0.5, 0.75],
                      regions=[(0.0, 0.25), (0.25, 0.75),
                               (0.75, 1.0), (0.0, 1.0)],
                      feature_list=None):
    """
    Input: Pandas.Series of strings;
           Transformers masked language model (e.g. BERT, RoBERTa, etc.);
           Transformers tokenizer;
           Device ('cpu' or 'gpu').
    Returns: 2D Numpy.Array of shape=(no. samples, (1 + 1 + no. quantiles) * 2 * no. input regions));
             mean, variance, and quantiles of the probability AND rank
               of the input sequence for different input regions.
    """
    if type(feature_list) == list:
        for i in range(len(regions)):
            feature_list.append('lm_proba_mean_region{}'.format(i))
            feature_list.append('lm_proba_var_region{}'.format(i))
            feature_list += ['lm_proba_quant{}_region{}'.format(j, i) for j in range(len(quantiles))]
            feature_list.append('lm_rank_mean_region{}'.format(i))
            feature_list.append('lm_rank_var_region{}'.format(i))
            feature_list += ['lm_rank_quant{}_region{}'.format(j, i) for j in range(len(quantiles))]

    start = time.time()

    # prepare data
    data = text_list.tolist()
    dataloader = DataLoader(data, sampler=SequentialSampler(data), batch_size=1)

    # no. tokens that can be fed into the model at one time
    assert hasattr(lm_masked_model.config, 'max_position_embeddings')
    max_length = lm_masked_model.config.max_position_embeddings - 2

    # compute language model outputs on a batch of examples
    feature_vec_list = []
    for i, text in enumerate(dataloader):

        # get no. words for this sample
        text = text[0].split()
        num_words = len(text)

        # compute features for each input region
        feature_vec = []
        for start_frac, end_frac in regions:

            # get input region text
            start_ndx = int(num_words * start_frac)
            end_ndx = int(num_words * end_frac)
            region_text = ' '.join(text[start_ndx:end_ndx])

            # get logits over all tokens in the vocabularly for each token in the sample
            # Note: don't need attention IDs since we are not truncating or padding anything
            region_ids = lm_masked_tokenizer(region_text, truncation=True,
                                             return_tensors='pt')['input_ids'][0].to(device)

            # compute logits in segments if no. tokens exceeds models max. input size
            if len(region_ids) > max_length:
                segment_logit_list = []
                for j in range(0, len(region_ids), max_length):
                    segment_ids = region_ids[j: j + max_length]
                    segment_logits = lm_masked_model(segment_ids.unsqueeze(dim=0), return_dict=True).logits[0]
                    segment_logit_list.append(segment_logits)
                region_logits = torch.vstack(segment_logit_list)

            # compute all logits at the same time
            else:
                region_logits = lm_masked_model(region_ids.unsqueeze(dim=0), return_dict=True).logits[0]

            # compute probability and rank of each token
            probas = region_logits.softmax(axis=1)
            ranks = probas.shape[1] - probas.argsort(axis=1).argsort(axis=1)

            # extract probability and rank of each token, high prob. == high rank (low no.)
            seq_proba = [probas[i][region_ids[i]].item() for i in range(len(region_ids))]
            seq_rank = [ranks[i][region_ids[i]].item() for i in range(len(region_ids))]

            # contruct feature vector for this sample
            feature_vec.append(np.mean(seq_proba))
            feature_vec.append(np.var(seq_proba))
            feature_vec += list(np.quantile(seq_proba, quantiles))
            feature_vec.append(np.mean(seq_rank))
            feature_vec.append(np.var(seq_rank))
            feature_vec += list(np.quantile(seq_rank, quantiles))

        # add feature vector to the list of feature vectors
        feature_vec_list.append(feature_vec)

        if logger and i % 100 == 0:
            s = '[LM: PROB. AND RANK] no. samples: {:,}...{:.3f}s'
            logger.info(s.format(i, time.time() - start))

    return np.vstack(feature_vec_list)


@apply_tags(['lm'])
def lm_perplexity(text_list, lm_causal_model, lm_causal_tokenizer, device,
                  logger=None, stride=1,
                  regions=[(0.0, 0.25), (0.25, 0.75),
                           (0.75, 1.0), (0.0, 1.0)],
                  feature_list=None):
    """
    Input: Pandas.Series of strings;
           Transformers causal lanugage model (i.e. predicts tokens sequentially) (e.g. GPT-2);
           Transformers tokenizer;
           Device ('cpu' or 'gpu').
    Returns: 2D Numpy.Array of shape=(no. samples, no. input regions));
             perplexity of the input sequence for different input regions.
    Reference: https://huggingface.co/transformers/perplexity.html
    """
    if type(feature_list) == list:
        feature_list += ['lm_perplexity_region{}'.format(i) for i in range(len(regions))]

    start = time.time()

    # prepare data
    data = text_list.tolist()
    dataloader = DataLoader(data, sampler=SequentialSampler(data), batch_size=1)

    # get max. input length of the model
    try:
        assert hasattr(lm_causal_model.config, 'n_positions')
        max_length = lm_causal_model.config.n_positions
    except:
        # you are probably using xlnet or other non-max-seq model
        max_length = 256

    # compute language model perplexity on each sample
    feature_vec_list = []
    for i, text in enumerate(dataloader):

        # get no. words in this sample
        text = text[0].split()
        num_words = len(text)

        # compute features for each input region
        feature_vec = []
        for start_frac, end_frac in regions:

            # get input region text
            start_ndx = int(num_words * start_frac)
            end_ndx = int(num_words * end_frac)
            region_text = ' '.join(text[start_ndx:end_ndx])

            # tokenize this sample
            region_ids = lm_causal_tokenizer(region_text, return_tensors='pt',
                                             truncation=True)['input_ids'][0].to(device)

            # set stride for the sliding window: computes perplexity in two steps
            num_steps = 2
            stride = min(int(len(region_ids) / num_steps) + 1, max_length)

            # compute perplexity of this sample
            lls = []
            for j in range(0, region_ids.size(0), stride):
                begin_loc = max(j + stride - max_length, 0)
                end_loc = min(j + stride, region_ids.size(0))
                trg_len = end_loc - j    # may be different from stride on last loop
                input_ids = region_ids[begin_loc:end_loc].to(device)
                target_ids = input_ids.clone()
                target_ids[:-trg_len] = -100

                with torch.no_grad():
                    try:
                        outputs = lm_causal_model(input_ids, labels=target_ids)
                    except:
                        outputs = lm_causal_model(input_ids.unsqueeze(0), labels=target_ids)
                    log_likelihood = outputs[0] * trg_len

                lls.append(log_likelihood)

            # handle safely if no elements in lls
            if len(lls) == 0:
                ppl = 0
            else:
                ppl = torch.exp(torch.stack(lls).sum() / end_loc).detach().cpu().item()

            # contruct feature vector for this sample
            feature_vec.append(ppl)

        # add feature vector to the list of feature vectors
        feature_vec_list.append(feature_vec)

        if logger and i % 100 == 0:
            s = '[LM: PERPLEXITY] no. samples: {:,}...{:.3f}s'
            logger.info(s.format(i + 1, time.time() - start))

    return np.vstack(feature_vec_list)

