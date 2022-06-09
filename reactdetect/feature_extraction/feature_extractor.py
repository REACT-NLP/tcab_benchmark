import logging
import time
import inspect
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from .extractors.utils import EXTR_FCNS_BY_TAG, EXTR_FCNS_BY_NAME
from .utils import save_extracted_samples


class FeatureExtractor:
    def __init__(self, add_tags=[], remove_tags=[], add_specific=[], remove_specific=[], logger=None):
        """
        Loads the functions with the provided tags, those functions named in the add_list and removes functions
        specified in remove_list.
        :param add_tags: List[str] a list of tags of extractor functions to include, e.g. if you only want
            text and language model extractor functions, add_tags=['tp', 'lm']
        :param remove_tags: List[str] a list of tags remove from, e.g. if you are donig classification
            you may want to drop functions with the 'seq2seq' tag, so remove_tags=['seq2seq']
        :param add_specific: List[str] a list of extractor function names to add in addition to those collected with the
            provided tags, e.g. ['tm_posterior', 'tm_gradient']
        :param remove_specific: List[str] a list of extractor function names to remove from set collected with the
            provided tags, e.g. ['tm_posterior', 'tm_gradient']
        :param logging_level: str 'info' or 'warn'
        """

        # create logger if not supplied
        if logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
        else:
            self.logger = logger

        # initially set vectorizer for tp_ngrams to None
        self.vectorizer = None

        # aggregate extractor functions by provided tags
        self.extractors = []
        for tag in add_tags:
            try:
                fcns_with_tag = EXTR_FCNS_BY_TAG[tag]
                fcns_with_tag.sort(key=lambda f: f.__name__)  # sort functions by name before adding
                self.extractors.extend(fcns_with_tag)
            except KeyError:
                raise KeyError(f'tag {tag} not found in EXTR_FCNS_BY_TAG dictionary')

        # remove functions by tag
        for tag in remove_tags:
            try:
                self.extractors = [fcn for fcn in self.extractors if fcn not in EXTR_FCNS_BY_TAG[tag]]
            except KeyError:
                raise KeyError(f'tag {tag} not found in EXTR_FCNS_BY_TAG dictionary')

        # add individual extractor functions by name, e.g. "tp_num_words"
        for fcn_name in add_specific:
            try:
                self.extractors.append(EXTR_FCNS_BY_NAME[fcn_name])
            except KeyError:
                raise KeyError(f'{fcn_name} is not registered in the EXTR_FCN_BY_NAME dictionary')

        self.extractors = list(dict.fromkeys(self.extractors))  # remove duplicate functions while preserving order

        # remove individual extractor functions by name, e.g. "tp_num_words"
        for fcn_name in remove_specific:
            try:
                self.extractors.remove(EXTR_FCNS_BY_NAME[fcn_name])
            except KeyError:
                raise KeyError(f'{fcn_name} is not registered in the EXTR_FCN_BY_NAME dictionary')
            except ValueError:
                self.logger.warning(f'{fcn_name} is not in self.extractors — no need to remove it; continuing...')

        # get all of the parameters across all of the feature extraction functions that don't have defaults
        self.necessary_params = []
        for fcn in self.extractors:
            for param in inspect.signature(fcn).parameters.values():
                if param.default is param.empty:
                    self.necessary_params.append(param.name)
        self.necessary_params = list(set(self.necessary_params))  # remove duplicate parameters
        self.logger.info(f'Non-default (required) parameters for extractor functions: {self.necessary_params}')
        self.logger.info(f'Functions to be run: {[fcn.__name__ for fcn in self.extractors]}')

        # sort self.extractors
        self.extractors = sorted(self.extractors, key=lambda f: f.__name__)

    def _fit_vectorizer(self, text_list, analyzer='char_wb', ngram_range=(3, 3), max_features=5000,
                        lowercase=False, use_idf=False):
        """Helper function for creating a vectorizer for use with the tp_ngrams extractor function.
        """
        vectorizer = TfidfVectorizer(stop_words='english',
                                     ngram_range=ngram_range,
                                     analyzer=analyzer,
                                     max_features=max_features,
                                     lowercase=lowercase,
                                     use_idf=use_idf)
        self.vectorizer = vectorizer.fit(text_list)
    
    def __call__(self, return_dict=False, clean_extremes=True, output_text_list=None, save_extracted=False, unused_model_to_cpu=False, **kwargs):
        """Calls each of the extractor functions, passing the appropriate
        parameters to each and aggregating the returned features.
        :param return_dict: bool, if False, aggregate all of the feature vectors from all of the extractor
            functions into a single large feature vector for each sample; otherwise, return a dictionary that
            is a mapping from extractor function names to the feature vectors for the sample
        :param clean_extremes: bool, if True, remove nans and very large values from the feature vectors
        :param output_text_list: List[str], if using a seq2seq dataset, you can pass the output text and get features
            for the output as well.
        :param save_extracted: bool, if True, save all of the extracted features for each sample after each
            extraction function has finished running
        """

        # preliminary check to make sure that all of the parameters that all
        # of the extraction functions require have been passed as keyword args
        assert set(self.necessary_params).issubset(list(kwargs.keys())), \
            f'Missing the following required params: {set(self.necessary_params) - set(list(kwargs.keys()))}'

        # check to make sure input and output text lists are the same length (only relevant for seq2seq tasks)
        if output_text_list is not None:
            assert len(kwargs['text_list']) == len(output_text_list), \
                f"text_list and output_text_list need to be the same length"

        # create containers for holding feature extraction outputs
        all_features = []  # list for holding all of the features collected from all extractor functions
        all_feature_names = []
        all_features_dict = {}  # a dictionary for holding mappings from extractor function names to extractor outputs

        # iterate through list of extractor functions
        self.logger.info(f'Starting extraction process...\n{"Function name":30}{"Runtime":>10}')
        for fcn in self.extractors:
            try:
                # ============= start setup for this extractor function =============

                if unused_model_to_cpu:
                    # move causal lm model to or from device
                    if fcn.__name__ == 'lm_perplexity':
                        kwargs['lm_causal_model'] = kwargs['lm_causal_model'].to(kwargs['device'])
                    elif 'lm_causal_model' in kwargs:
                        kwargs['lm_causal_model'] = kwargs['lm_causal_model'].cpu()

                    # move masked lm model to or from device
                    if fcn.__name__ == 'lm_proba_and_rank':
                        kwargs['lm_masked_model'] = kwargs['lm_masked_model'].to(kwargs['device'])
                    elif 'lm_masked_model' in kwargs:
                        kwargs['lm_masked_model'] = kwargs['lm_masked_model'].cpu()

                    # move masked lm model to or from device
                    if fcn.__name__ == 'lm_bert':
                        kwargs['lm_bert_model'] = kwargs['lm_bert_model'].to(kwargs['device'])
                    elif 'lm_bert_model' in kwargs:
                        kwargs['lm_bert_model'] = kwargs['lm_bert_model'].cpu()
                    
                    # move masked lm model to or from device
                    if fcn.__name__ == 'tp_bert':
                        kwargs['lm_bert_model'] = kwargs['lm_bert_model'].to(kwargs['device'])
                    elif 'lm_bert_model' in kwargs:
                        kwargs['lm_bert_model'] = kwargs['lm_bert_model'].cpu()

                    # move target model to or from device
                    if fcn in EXTR_FCNS_BY_TAG['tm']:
                        kwargs['target_model'] = kwargs['target_model'].to(kwargs['device'])
                    elif 'target_model' in kwargs:
                        kwargs['target_model'] = kwargs['target_model'].cpu()

                # get all parameters for this function
                fcn_params = [param.name for param in inspect.signature(fcn).parameters.values()]

                # get subset of kwargs specifically for this function
                kwargs_ = {k: kwargs[k] for k in fcn_params if k in kwargs}

                # for tp_ngrams, if fitted vectorizer not passed OR it's already fit, fit it on the supplied text_list
                if fcn.__name__ == 'tp_ngrams' and \
                        'vectorizer' not in kwargs_ and \
                        self.vectorizer is None:
                    self._fit_vectorizer(**kwargs_)
                    kwargs_['vectorizer'] = self.vectorizer

                # pass logger to extractor function if it accepts one
                if 'logger' in fcn_params:
                    kwargs_['logger'] = self.logger

                # ============= end setup for this extractor function =============

                # list for holding the NAMES of the features being collected for this extractor function
                feature_names = []

                # run the extractor function
                start = time.time()
                features = fcn(feature_list=feature_names, **kwargs_)  # pass the subset, kwargs_, to the extractor function
                self.logger.info(f'{fcn.__name__:30}{time.time() - start:>10.3f}s ')

                # run extractor function for output text if provided and fcn is designed to create features for output
                if output_text_list is not None and fcn in EXTR_FCNS_BY_TAG['output_capable']:
                    kwargs_['text_list'] = output_text_list  # swap out text_list for output_text_list
                    output_feature_names = []

                    start = time.time()
                    output_features = fcn(feature_list=output_feature_names, **kwargs_)
                    self.logger.info(f'{"    (output text)":30}{time.time() - start:>10.3f}s ')

                    # add '_output' string to feature names to help differentiate
                    for i in range(len(output_feature_names)):
                        output_feature_names[i] += '_output'

                    # add to features for input
                    feature_names.extend(output_feature_names)
                    features = np.hstack([features, output_features])

                # if clean_extremes == True: remove nans, infinities, and numbers that are very large
                if clean_extremes:
                    features = np.clip(np.nan_to_num(features), a_min=-1e7, a_max=1e7)

                # place output of function in containers
                all_features.append(features)
                all_feature_names.extend(feature_names)
                all_features_dict[fcn.__name__] = (feature_names, features)

                # save what has been extracted thus far
                if save_extracted:
                    assert 'df' in kwargs, 'need to provide df if saving extracted samples'
                    assert 'save_path' in kwargs, 'need to provide path to save extracted samples'
                    save_extracted_samples(all_features_dict, kwargs['text_list'], kwargs['df'], kwargs['save_path'])

            except Exception as e:
                # self.logger.exception(f'Extractor function: {fcn.__name__} FAILED', str(e))
                print(f'Extractor function: {fcn.__name__} FAILED', str(e))
                raise e

        # if return_dict is False, merge all feature extractor outputs into a single feature vector for each sample
        if not return_dict:
            all_features = np.hstack(all_features)

            return all_features, all_feature_names

        # if return_dict is True, return the dictionary of features without aggregating into single feature vector
        else:
            return all_features_dict


if __name__ == '__main__':

    # load feature extractor
    fe = FeatureExtractor(
        add_tags=['tp'],  # get all of the feature extraction functions with the 'tp' tag
        remove_tags=['seq2seq'],  # remove functions with these tags
        remove_specific=['tp_ngrams']  # remove the 'tp_num_words' function
    )
    # get samples
    text_list = pd.Series(['hello there h0w are yuo.', 'h3llo there how are you.'])  # attacked samples

    # call the feature extractor with the required arguments
    features, names = fe(text_list=text_list)

    print(features.shape)  # shape should be (2, 36)
