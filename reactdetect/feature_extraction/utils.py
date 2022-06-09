import os
from pathlib import Path
import torch
import pandas as pd


def save_extracted_samples(feature_dict: dict, text_list: list, df_: pd.DataFrame, path: str):
    """Saves the features in feature_dict for each sample using the pandas DataFrame df to
    get the samples' global indices.
    :param feature_dict: a dictionary where the keys are extractor function names and the
        values are tuples containing feature names and extracted feature values
    :param text_list: the list of text_samples that was passed to the FeatureExtractor
    :param df_: the dataframe from which the text_list samples came from; i.e.
        list(df['perturbed_text']) should be equal to text_list; this dataframe should have
        a 'global_ndx' column that will be used to map samples to their extracted feature files.
    :param path: the file path to the directory in which to create the folder that will contain
        the extracted samples
    """

    out_dir = os.path.join(path, 'extracted_samples')
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # reset index so that indices can align with text_list indices
    df_ = df_.reset_index(drop=True)

    # get extracted features for each sample in text_list
    for i in range(len(text_list)):

        # a dictionary for holding all of the extracted features for sample i
        sample_i = {}

        # for each extractor function, get the extracted features for sample i
        for fcn_name in feature_dict:
            feature_names, all_features = feature_dict[fcn_name]
            sample_i[fcn_name] = (feature_names, all_features[i])

        # get information about this sample
        assert df_.perturbed_text[i] == text_list[i]  # always make sure text list and dataframe are aligned
        global_ndx = df_.global_ndx[i]  # get "global" index in attack dataset for this sample
        sample_i['_sample_info'] = dict(df_.iloc[i])  # also save row of attack dataset for sample
        sample_i_out_dir = os.path.join(out_dir, f'sample_{global_ndx}.pt')

        # if sample already has file for extracted features, update file rather than overwrite
        if os.path.isfile(sample_i_out_dir):
            prev_sample_i = torch.load(sample_i_out_dir)  # load the previously saved features for sample i
            prev_sample_i.update(sample_i)  # update the previously saved sample's features with the new features
            torch.save(prev_sample_i, sample_i_out_dir)
        else:
            torch.save(sample_i, sample_i_out_dir)