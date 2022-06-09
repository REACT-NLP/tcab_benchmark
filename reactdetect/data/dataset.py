from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as train_test_split_skl
from collections import Counter
import math


class ReactDataset:
    def __init__(self,
                 file_dir: str = None,
                 df: pd.DataFrame = None,
                 verbose: bool = False):
        """Takes either: (1) a path to a CSV file with the REACT style columns or (2) a pandas DataFrame with
        the REACT style columns and creates a wrapper around it that can interact well with AllenNLP.
        :param file_dir: the path to the CSV file that has a column for 'perturbed_text', a column for
            'perturbed_output', a column for 'attack_name', and a column for 'original_text';
            if seq2seq is True, there should also be a column for 'original_output'; the CSV file should have a header.
        :param df: a pandas DataFrame with the columns
        :param verbose: if True, all the vprint() will be activated
        """

        # if path to CSV file is provided, read from CSV into dataframe using pandas
        if file_dir is not None:
            self.df = pd.read_csv(file_dir, low_memory=False)
        else:
            assert df is not None, 'either a path to a CSV file, or a pd.DataFrame object must be provided'
            self.df = df

        # create mapping for attack names; e.g. {0: 'malicious nonsense', 1: 'universal dropper'}
        
        self.label_mapping = {attack: i for i, attack in enumerate(self.df['attack_name'].unique())}

        
        

    def build_dataset(self,
                      setting: str = 'multiclass_with_clean',
                      attacks_considered: List[str] = [],
                      oversample: bool = False,
                      shuffle_data: bool = True,
                      max_instance_per_class = math.inf,
                      min_instance_per_class = 0,
                      random_seed: int = 1):
        """Builds dataset according to arguments provided from a results.csv file.
        :param setting: the setting for this attack; there are currently 4 settings offered:
            (1) 'clean_vs_attack' produces a dataset with two classes, clean and attack, where all of the
                attacked samples are produced by the same attacker (so attacks_considered should only contain
                the name of a single attack.
            (2) 'clean_vs_multi_attack' produces a dataset with two classes, clean and attack, where the attacked
                samples are a mixture of all of the attacks in the attacks_considered list.
            (3) 'multiclass_with_clean' produces a dataset with k+1 classes, one for each of the k attacks listed in the
                attacks_considered list plus a class for clean samples.
            (4) 'multiclass_without_clean' produces a dataset with k classes, one for each of the k attacks listed in the
                attacks_considered list; clean samples are not included.
        :param attacks_considered: a list of the attacks to include when building the dataset for this setting.
        :param oversample: if True, the classes in this dataset will be balanced by oversampling the classes that have
            lower numbers of samples until the number of samples per class is constant across all classes
        :param shuffle_data: whether the data should be shuffled or not
        :param random_seed: random seed to control randomness
        """

        if attacks_considered == 'all': #wildcard
            attacks_considered = self.get_available_attacks()

        # check that dataframe has the required additional columns
        assert set(['test_ndx', 'dataset']).issubset(self.df.columns), \
            'to use the build_dataset method, self.df must have a "test_ndx" column and a "dataset" column'

        # filter out attacks that aren't being considered
        df = self.df[self.df['attack_name'].isin(attacks_considered)]

        # choose which clean samples will be included based on TP, FP, TN, FN rates;
        # only use when target model's task is a binary classification problem
        if 'original' in df['attack_name'].unique() and not self.seq2seq \
                and 'label_actual' in df.columns and 'label_before' in df.columns:
            df = select_clean_samples(df, setting, random_seed=random_seed)

        # verify dataframe is okay given setting and then filter dataframe
        df = filter_by_setting(df, setting)

        # TODO: add better support for seq2seq datasets
        self.df, self.label_mapping = distribute_samples(df, oversample=oversample, \
        max_instance_per_class=max_instance_per_class,min_instance_per_class=min_instance_per_class,random_seed=random_seed)

        if shuffle_data:
            self.df = self.df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        return self


    def get_samples(self) -> List[str]:
        """
        Return a copy of all the text samples stored in the class.
        """

        # get input text
        try:
            text_list = list(self.df['text'])  # TODO: clean up this
        except KeyError:
            text_list = list(self.df['perturbed_text'])

        return text_list

    def get_samples_outputs(self) -> List[str]:
        """
        Returns a copy of the outputs associated with each sample for seq2seq datasets.
        """
        return list(self.df['perturbed_output'])

    def get_labels(self) -> List[str]:
        """
        Return a copy of all the labels stored in the class.
        """
        try:
            return list(self.df['label'])  # TODO: clean this up
        except KeyError:
            return list(self.df['attack_name'])

    def get_label_mapping(self) -> Dict:
        """
        Return a dictionary of label->integer mapping.
        The mapping was automatically constructed in the initializer.
        """
        return self.label_mapping

    def get_available_classes(self) -> List[str]:
        """
        Return all avail classes, clean included
        """
        return list(Counter(self.df['attack_name']).keys())

    def get_available_attacks(self) -> List[str]:
        """
        Return all avail classes, clean excluded
        """
        return [i for i in self.get_available_classes() if i != 'original']

    def get_prediction_mapping(self) -> Dict:
        """
        Return a dictionary of text->integer prediction for target model mapping.
        """
        assert('label_after' in self.df.columns)
        out = {}
        for i,row in self.df.iterrows():
            out[row['perturbed_text']] = row['label_after']
        return out

    def __str__(self) -> str:
        return 'ReactDataset: '+ str(Counter(self.df['attack_name']))

    def __repr__(self) -> str:
        return str(self)

    @property
    def num_samples(self):
        return len(self.df)

    def set_verbose(self, verbose:bool):
        self.verbose=verbose

    def vprint(self, *args, **kw):
        """
        print function that can be tuned off with self.verbose flag.
            should've just use logger, probably..
        """
        if self.verbose:
            print(*args, **kw)

    def train_test_split(self, test_size=0.2, shuffle=True, random_seed=1):
        X = np.array(self.get_samples(), dtype=object)
        y = np.array(self.get_labels(), dtype=object)
        x_train, x_test, y_train, y_test = train_test_split_skl(X, y, test_size=0.2, random_state=random_seed, shuffle=shuffle, stratify=y)
        return x_train, x_test, y_train, y_test


################################################
# Brophy's utility functions for preprocessing #
################################################

def select_clean_samples(attack_df: pd.DataFrame,
                         setting: str,
                         random_seed: int = 1):
    """
    Selects a subset of original samples proportional to the no. of original samples
    from each distribution: TN, FP, FN, TP, for each dataset. Used for binary classification datasets.
    :param attack_df: the dataframe with for 'attack_name', 'test_ndx', 'label_actual', and 'label_before'
    :param setting: 'clean_vs_attack', 'clean_vs_multi_attack', 'multiclass_with_clean', or 'multiclass_without_clean'
    :param random_seed: random seed used by sklearn's train_test_split function
    """
    label_map = {0: 'TN', 1: 'FP', 2: 'FN', 3: 'TP'}

    def pred_type(x, y):
        """
        Encodes true label "x" and pred "y" into one of the following:
        TN (0), FP (1), FN (2), or TP (3).
        """
        assert x in [0, 1] and y in [0, 1]

        if x == 0:
            result = 0 if y == 0 else 1

        elif x == 1:
            result = 2 if y == 0 else 3

        return result

    # focus on one dataset at a time
    df = attack_df.copy()

    # get attack samples for this dataset
    temp1_df = df[df['attack_name'] != 'original']

    # make sure there are some samples
    assert len(temp1_df) != 0

    # get sum of samples over all attacks considered
    if setting == 'clean_vs_multi_attack':
        num_attack_samples = len(temp1_df['test_ndx'].unique())

    # get avg. no. of samples over all attacks considered
    else:
        num_attack_samples = int(len(temp1_df['test_ndx'].unique()) / len(temp1_df['attack_name'].unique()))

    # compute confusion matrix
    temp2_df = df[df['attack_name'] == 'original'].copy()
    temp2_df['pred_type'] = temp2_df.apply(lambda x: pred_type(x.label_actual, x.label_before), axis=1)
    temp2_df['pred_type_label'] = temp2_df['pred_type'].apply(lambda x: label_map[x])

    # get number of clean samples
    num_clean_samples = min(num_attack_samples, len(temp2_df) - 4)  # TN, FP, FN, and TN

    # select a stratified subset of original samples
    _, clean_df = train_test_split(temp2_df, test_size=num_clean_samples,
                                   stratify=temp2_df['pred_type'], random_state=random_seed)
    clean_df = clean_df.drop(['pred_type', 'pred_type_label'], axis=1)

    # assemble new attack dataset
    attack_df = pd.concat([clean_df, temp1_df])

    return attack_df


def filter_by_setting(attack_df: pd.DataFrame,
                      setting: str):
    """
    Verifies the setting matches the attacks considered, and then
    adds clean samples if necessary.
    :param attack_df: a dataframe with at least an 'attack_name' column
    :param setting: 'clean_vs_attack', 'clean_vs_multi_attack', 'multiclass_with_clean', or 'multiclass_without_clean'
    """
    df = attack_df.copy()

    # get list of attacks in the dataset
    attack_list = df['attack_name'].unique()

    # adjust dataset to match setting
    if setting == 'clean_vs_attack':
        assert len(attack_list) == 2 , 'you need 2 attacks only'
        assert 'clean' in attack_list, 'need to include clean!'

    elif setting == 'clean_vs_multi_attack':
        assert len(attack_list) > 2 , 'you need > 2 attacks !'
        assert 'clean' in attack_list, 'you need  clean'
        df['attack_name'] = df['attack_name'].apply(lambda x: 'clean' if x == 'clean' else 'perturbed')

    elif setting == 'multiclass_without_clean':
        assert len(attack_list) >= 2 , 'you need >= 2 attacks!'
        assert 'clean' not in attack_list, 'you need no clean!'
        df = df[df['attack_name'] != 'clean']

    elif setting == 'multiclass_with_clean':
        assert len(attack_list) >= 2 ,'you need >2 attacks!'
        assert 'clean' in attack_list, 'you need  clean!'

    else:
        raise ValueError('setting {} unknown!'.format(setting))

    return df


def distribute_samples(in_df: pd.DataFrame,
                       oversample: bool = False,
                       max_instance_per_class = math.inf,
                       min_instance_per_class = 0,
                       random_seed: int = 1):
    """
    Distribute samples to each attack, making sure
    not to use the same sample twice.
    :param in_df: the dataframe with for 'attack_name', 'test_ndx', 'label_actual', and 'label_before'
    <**** WARNING NEVER CALL THIS ****>:param oversample: if True, the classes in this dataset will be balanced by oversampling the classes that have
            lower numbers of samples until the number of samples per class is constant across all classes <**** WARNING NEVER CALL THIS ****>
    :param random_seed: random seed used by numpy when distributing samples to different classes
    """

    # make a copy of the dataset
    df = in_df.copy()

    # shuffle and drop duplicate test indices from the same attack,
    # really only in problem in the clean vs. multiple attacks setting
    df = df.sample(frac=1.0, random_state=random_seed)
    
    df = df.drop_duplicates(['target_model_dataset', 'attack_name', 'test_index','target_model'])
    

    # for now do not reset idx
    from reactdetect.utils.pandas_ops import no_duplicate_index 
    assert (no_duplicate_index(df))

    # create unique indices
    # df = df.reset_index()

    # create mapping for attack names
    # (zhouhanx): putting sorted back to anchor label_map mapping behavior
    label_map = {i:attack  for i, attack in enumerate(sorted(df['attack_name'].unique()))}

    # make sure clean is always index 0, swapping if necessary
    for k in label_map.keys():
        if label_map[k] == 'original' and k != 0:
            label_map[k] = label_map[0]
            label_map[0] = 'original'

    # assign indices for each unique sample
    df['index'] = df.groupby(['target_model_dataset', 'test_index']).ngroup()
    

    # assign set of available indices for each attack
    sets_list = {attack: set(df[df['attack_name'] == attack]['index']) for attack in label_map.values()}

    # filtered attack set containers
    new_sets_list = {attack: set() for attack in sets_list.keys()}

    # initialize a random number generator
    rng = np.random.default_rng(random_seed)

    # continue assigning samples until the pool of samples is empty
    num_samples = len(df)
    while num_samples > 0:

        # assign one sample per attack
        for attack in label_map.values():
            ndx = None

            # choose a random sample if this attack's pool is not empty
            if len(sets_list[attack]) > 0:
                ndx = rng.choice(list(sets_list[attack]))
                new_sets_list[attack].add(ndx)
                num_samples -= 1

            # remove chosen sample from all attack sets
            if ndx is not None:
                for inner_attack in sets_list.keys():
                    if ndx in sets_list[inner_attack]:
                        sets_list[inner_attack].remove(ndx)

            # count number of remaining samples
            num_samples = sum(len(v) for k, v in sets_list.items())

            # check to see if all samples have been assigned
            if num_samples == 0:
                break

    # get data for each attack using the assigned samples
    dfs = []
    n_samples_per_class = []  # holds the number of samples for each attack
    for attack, sample_indices in new_sets_list.items():
        gf = df[(df['attack_name'] == attack) & (df['index'].isin(sample_indices))].copy()
        gf['label'] = attack
        gf['text'] = gf['perturbed_text']
        # only consider classes that have >= certain number of instances
        if len(gf) >= min_instance_per_class:
            # downsample each class's dataframe untill add of the classes are <= maximum samples per class
            if len(gf) > max_instance_per_class:
                gf = gf.sample(n=max_instance_per_class, random_state=random_seed)
            n_samples_per_class.append(len(gf))
            dfs.append(gf)

    # drop classes that contains less instance than minimum required
    dfs = [_ for _ in dfs if len(_)>min_instance_per_class]
    
    # oversamples each class's dataframe until all of the classes have the same number of samples as the
    # class that originally had the most samples
    if oversample:
        print('*** WARNING VERY IMPORTANT: [[[NEVER]]] call oversample via [[[build_dataset]]]   ***')
        print('***     the param will {leak entries} in train data to dev, the param is left here ***')
        print('***     so that legacy code won t break immediately                               ***')
        print('***     consider using the new .train_test_split method                           ***')
        print('***     use the following semantics:                                               ***')
        print('>>> x_train, x_test, y_train, y_test = dataset.train_test_split(test_size=0.2)')
        print('>>> train_loader = ReactDataLoader(x_train, y_train).oversample()')
        print('***     or alternatively                                                          ***')
        print('>>> train_loader = ReactDataLoader(x_train, y_train, oversample=True)')
        print('*** END WARNING                                                                   ***')
        raise AttributeError

    df = pd.concat(dfs)

    return df, {v: k for k, v in label_map.items()}


if __name__ == '__main__':
    dataset = ReactDataset('stub.csv', seq2seq=True)
    dataset.build_dataset(setting='multiclass_with_clean',
                          attacks_considered=['original', 'malicious nonsense', 'universal dropper'],
                          oversample=True)
