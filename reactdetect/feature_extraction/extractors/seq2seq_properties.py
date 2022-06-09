import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler
import bert_score
from .utils import apply_tags, split_sentences
import sys
from scipy.special import softmax
from .model_wrappers import NodeActivationWrapper, SaliencyWrapper


@apply_tags(['lm', 'seq2seq'])
def lm_bert_score(text_list, labels, device, batch_size=64, agg_types=['min', 'max'], feature_list=None):
    if type(feature_list) == list:
        feature_list.append('lm_bert_score')

    full_input_list = []
    full_output_list = []
    break_idxs = []

    # make sure input and output lists have the same number of samples
    assert len(text_list) == len(labels)

    # build input for bert score function so that things can be nicely batched
    for i in range(len(text_list)):
        # split this source into sentences
        source_i_sentences = split_sentences(text_list[i])

        # add all of these source sentences to the full list
        full_input_list.extend(source_i_sentences)

        # add as many copies of the output as there are sentences in the source
        full_output_list.extend([labels[i]] * len(source_i_sentences))

        assert len(full_input_list) == len(full_output_list)

        # record which index this sample ends on
        break_idxs.append(len(full_output_list) + 1)

    # calculate bert_score for all pairs of input sentence and output
    all_preds, hash_code = bert_score.score(
        full_input_list,
        full_output_list,
        model_type=None,
        num_layers=1,
        verbose=False,
        idf=False,
        device=device,
        batch_size=batch_size,
        lang='en',
        return_hash=True,
        rescale_with_baseline=False,
        baseline_path=None,
    )

    F1_scores = all_preds[2]

    # aggregate similarity scores for each sample
    all_sim_scores = []
    left_idx = 0
    for right_idx in break_idxs:

        # get the similarity scores for this sample
        sample_sims = F1_scores[left_idx: right_idx]

        # aggregate across the similarity scores in various ways
        sample_sims_agg = []
        if len(sample_sims) == 0:
            sample_sims_agg.extend([-1] * len(agg_types))
        else:
            if 'min' in agg_types:
                sample_sims_agg.append(min(sample_sims))
            if 'max' in agg_types:
                sample_sims_agg.append(max(sample_sims))
            if 'avg' in agg_types:
                sample_sims_agg.append(sum(sample_sims) / len(sample_sims))

        all_sim_scores.append(sample_sims_agg)

        # reset left_idx in order to move from left to right across F1_scores
        left_idx = right_idx

    return np.vstack(all_sim_scores)

@apply_tags(['tm', 'seq2seq'])
def tm_posterior_seq2seq(text_list, labels, target_model, batch_size=10, num_regions=4, logger=None, feature_list=None):
    """ This function is different from the classification version in that it aggregates across the vocab dimension
    in the output space. This is necessary because the output is too large otherwise.
    Input: text_list: List[str] a list of the input text sample to get the posteriors for
           target_model: a seq2seq model that can return logits when supplied with list of text input
           num_regions: the number of regions to break the logits into (must do this because
                unlike for classification models, this is very large, so we need to aggregate)
           logger: a logging object for updating user on this function's progress
           feature_list: the list of features to append `tm_gradient` to
    """
    if type(feature_list) == list:
        feature_list.append('tm_output')

    # start timer
    start = time.time()

    # create dataloader
    data = text_list.tolist()
    labels = labels.tolist()
    dataloader = DataLoader(data, sampler=SequentialSampler(data), batch_size=batch_size)

    # set to eval mode
    target_model.model.eval()

    # generate predictions on the test set
    all_preds = []
    for step, batch_text_list in enumerate(dataloader):

        with torch.no_grad():
            # make predictions for this batch
            preds = target_model.logits(batch_text_list, labels)  # shape = (batch sz, max len, vocab sz)

            # break into list in order to compute stats for each sample's output separately
            preds = preds.split(1, dim=0)

            # get stats for each sample
            for pred in preds:

                # break each sample into regions and compute stats on region separately
                pred_regions = torch.split(pred, int(pred.shape[1] / num_regions), dim=1)

                pred_stats = []
                for pred_region in pred_regions:
                    # this is a problem when each region only holds one word; this can happen when num_regions
                    # is too small relative the number of words/tokens in the input
                    pred_region = pred_region.squeeze(dim=0)

                    # compute stats for this region
                    means = pred_region.mean(dim=-1)
                    vars = pred_region.var(dim=-1)
                    maxes = pred_region.max(dim=-1)[0]
                    mins = pred_region.min(dim=-1)[0]

                    # get difference between largest and second largest logit values for each position in output
                    top2s = torch.topk(pred_region, 2, dim=-1)[0]
                    diffs = top2s[:, 0] - top2s[:, 1]

                    # merge all stats for this region
                    stats = torch.cat([means, vars, maxes, mins, diffs], dim=0)  # shape = (5, 1024)
                    stats = stats.flatten()

                    # merge stats for this region with other regions' stats
                    pred_stats.append(stats)

                pred_stats = torch.cat(pred_stats, dim=0).flatten()
                all_preds.append(pred_stats.cpu().numpy().tolist())

        # progress update
        if logger:
            s = '[TM: POSTERIOR] batch {:,} no. samples: {:,}'
            logger.info(s.format(step + 1, batch_size * (step + 1)))

    # concat all predictions
    all_preds = np.vstack(all_preds)

    return all_preds

@apply_tags(['tm', 'seq2seq'])
def tm_gradient_seq2seq(text_list, labels, target_model, num_regions=4, logger=None, feature_list=None):
    """ This function is different from the `tm_gradient` function for classification in the way in which
    the gradient is calculated. Specifically, the gradient of the maximum logits of the target model wrt
    the input layer rather than the gradient of the loss (which is produced using true labels).
    Input: text_list: List[str] list of text inputs to get gradient wrt
           target_model: a seq2seq model that has a method `gradient` that
                returns the gradient of the model wrt supplied input
           num_regions: the number of regions to break the gradients into (must do this because
                the size of the gradients vector is very large, so we need to aggregate)
           logger: a logging object for updating user on this function's progress
           feature_list: the list of features to append `tm_gradient` to
    """

    # # operate on the wrapped model
    # target_model = target_model.model

    # timer
    start = time.time()

    if type(feature_list) == list:
        feature_list.append('tm_gradient')

    # create dataloader
    data = list(zip(text_list.tolist(), labels.values))
    dataloader = DataLoader(data, sampler=SequentialSampler(data), batch_size=1)

    # activate evaluation mode
    target_model.eval()

    # compute loss for each sample
    feature_vec_list = []
    for i, (text, label) in enumerate(dataloader):

        # reset gradients of model to zero and extract gradients for this sample
        all_gradients = target_model.gradient(text, label).cpu()  # shape = (vocab sz, input shape)

        # break into regions
        all_gradients = torch.split(all_gradients, int(all_gradients.shape[1] / num_regions), dim=1)

        # calculate stats for each region
        all_stats = []
        for gradient in all_gradients:

            # get mean, variance, max, and min logit values across vocab for each position in output (for each region)
            means = gradient.mean(dim=0, keepdim=True)
            vars = gradient.var(dim=0, keepdim=True)
            maxes = gradient.max(dim=0, keepdim=True)[0]
            mins = gradient.min(dim=0, keepdim=True)[0]

            # merge all stats for this region
            stats = torch.cat([means, vars, maxes, mins], dim=0)
            stats = stats.flatten()

            # add stats for this region to stats for all regions of this sample
            all_stats.append(stats)
        all_stats = torch.cat(all_stats, dim=0)  # shape = (4, 1024)

        feature_vec_list.append(all_stats.flatten().cpu().numpy().tolist())

        # progress update
        if logger and i % 100 == 0:
            s = '[TM: GRADIENT] no. samples: {:,}...{:.3f}s'
            logger.info(s.format(i + 1, time.time() - start))

    return np.vstack(feature_vec_list)