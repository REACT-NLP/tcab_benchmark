import shutil
import resource
from collections import Counter
from tkinter.font import names

import numpy as np

from make_offical_dataset_splits import create_ideal_train_test_split
from reactdetect.utils.device_casting import move_to_cpu
from reactdetect.allennlp_wrappers import ReactDataLoader
from reactdetect.utils.file_io import grab_joblibs, load_json, path_splitall
from reactdetect.feature_extraction.feature_extractor import FeatureExtractor
from reactdetect.aggregation.label_embedders import ReactLabelEmbedder
from reactdetect.aggregation.feature_embedders import ReactSampleWiseFeatureEmbedder
from reactdetect.allennlp_wrappers import ReactDataLoader
from reactdetect.models.base_models import ReactClassicalModel
from reactdetect.allennlp_wrappers import ReactClassicalModelTrainer
from reactdetect.utils.file_io import mkdir_if_dne, rm_rf_dir, grab_joblibs, vim_write_zz
from reactdetect.utils.feature_dim_names import get_feature_dim_names
from reactdetect.utils.magic_vars import SUPPORTED_TARGET_MODELS, SUPPORTED_TARGET_MODEL_DATASETS
from reactdetect.utils.device_casting import move_to_cpu

from sklearn import preprocessing
from sklearn.model_selection import train_test_split as train_test_split_skl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle as sk_shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from lightgbm import LGBMClassifier
from bertdetector_repr import load_bert_detector, re_encode_with_bertclassifier

import joblib
import numpy as np
import os
from collections import Counter
import sys


def clear_dir(in_dir):
    """
    Clear contents of directory.
    """
    if not os.path.exists(in_dir):
        return -1

    # remove contents of the directory
    for fn in os.listdir(in_dir):
        fp = os.path.join(in_dir, fn)

        # directory
        if os.path.isdir(fp):
            shutil.rmtree(fp)

        # file
        else:
            os.remove(fp)

    return 0


class StubEmbedderSamplewise(ReactSampleWiseFeatureEmbedder):
    def __init__(self, feature_names, verbose=False):
        super().__init__(use_cache=False, cuda_device=None, verbose=False)
        self.feature_names = feature_names
    def embed_texts(self, X):
        "do nothing"
        return X
    def get_feature_names(self):
        return self.feature_names

def get_feature_names_by_setting(feature_setting):
    if feature_setting == 'bert':
        feature_names = ['tp_bert']
    elif feature_setting == 'bert+tp':
        feature_names  = sorted([fcn.__name__ for fcn in FeatureExtractor(add_tags=['tp']).extractors])
    elif feature_setting == 'bert+tp+lm':
        feature_names  = sorted([fcn.__name__ for fcn in FeatureExtractor(add_tags=['lm','tp']).extractors])
    elif feature_setting == 'bert+tp+lm+tm' or feature_setting=='all':
        feature_names  = sorted([fcn.__name__ for fcn in FeatureExtractor(add_tags=['lm','tp','tm']).extractors])
    else:
        print('invalid feature combo! must be bert / bert+tp / bert+tp+lm/ bert+tp+lm+tm / all(equals bert+tp+lm+tm)!')
        raise NameError
    #print('you will using the following features \n', feature_names,'\n')
    return feature_names

def get_feature_dim_names_by_feature_names(feature_names):
    out = []
    for _fname in feature_names:
        out += get_feature_dim_names(_fname)
    return out

def joblib_to_x_and_y(joblib_dir, feature_names, mask_to_binary, args):
    print('--- loading from ', joblib_dir, 'mask to bin? ', mask_to_binary)
    valid_features=feature_names
    instance_dict = joblib.load(joblib_dir)
    if args.detection_bert_root != None:
        bert_detector = load_bert_detector(args.detection_bert_root)
        print('loader bert detector')
        instance_dict = re_encode_with_bertclassifier(instance_dict, bert_detector)
        print('done re encoding instance_dict')
        del bert_detector

    print('num of instances in the file: ', len(instance_dict))
    out_x = []
    out_y = []
    for k in instance_dict:
        instance = instance_dict[k]
        feats = []
        for f in valid_features:
            feat = move_to_cpu(instance['deliverable'][f])
            feats.append(feat)
        
        feats = np.concatenate([i[0] for i in feats])
        out_x.append(feats)
        if mask_to_binary:
            if k[0] == 'clean':
                out_y.append('clean')
            else:
                out_y.append('perturbed')         
        else: # mask to binary
            out_y.append(k[0])
        unused_feats = [f for f in instance['deliverable'].keys() if f not in valid_features]
        for uf in unused_feats:
            del instance['deliverable'][uf] #save memory
    del instance_dict
    print('loaded, labels: ',Counter(out_y))
    assert(len(out_x) == len(out_y))
    return out_x, out_y


def run_experiment(experiment_root_dir, feature_setting, setting, model, test_mode, args):

    if setting == 'clean_vs_all':
        print('--- your labels will be masked to binary')
        MASK_TO_BIN = True 
    else:
        MASK_TO_BIN = False

    all_jblbs = grab_joblibs(experiment_root_dir)
    names_of_feats_to_use = get_feature_names_by_setting(feature_setting) 
    print('--- you will use these features')
    print(names_of_feats_to_use)
    if model == 'lr' or model == 'LR':
        output_dir = os.path.join(experiment_root_dir, 'LR', feature_setting)
    elif model == 'lgb' or model == 'LGB':
        output_dir =  os.path.join(experiment_root_dir, 'LGB', feature_setting)
    elif model == 'rf' or model == 'RF':
        output_dir =  os.path.join(experiment_root_dir, 'RF', feature_setting)
    elif model == 'dt' or model == 'DT':
        output_dir =  os.path.join(experiment_root_dir, 'DT', feature_setting)
    else:
        print('model must be lr/lgb/rf/dt!')
        raise TypeError
    if args.detection_bert_root != None:
        output_dir += '_reencode_with'
        output_dir = os.path.join(
                    output_dir,
                    '@'.join(path_splitall(args.detection_bert_root)).strip('.').strip()
                )
    print('--output to ', output_dir)
    sample_wise_embedder = StubEmbedderSamplewise(
        feature_names = get_feature_dim_names_by_feature_names(names_of_feats_to_use)
    )

    train_dat_dir = os.path.join(experiment_root_dir, 'train.joblib')
    val_dat_dir = os.path.join(experiment_root_dir, 'val.joblib')
    test_dat_dir = os.path.join(experiment_root_dir, 'test_release.joblib')
    whole_dat_dir = os.path.join(experiment_root_dir, 'data.joblib')
    
    if os.path.join(experiment_root_dir, 'train.joblib') in all_jblbs and os.path.join(experiment_root_dir, 'val.joblib') and os.path.join(experiment_root_dir, 'test_release.joblib')in all_jblbs:
        print('--- looks you have a pre-splitted train-val-test(release) split, potentially due to ablation analysis purpose')
        x_train, y_train = joblib_to_x_and_y( os.path.join(experiment_root_dir, 'train.joblib'), names_of_feats_to_use , MASK_TO_BIN, args)
        x_val, y_val = joblib_to_x_and_y( os.path.join(experiment_root_dir, 'val.joblib'), names_of_feats_to_use, MASK_TO_BIN, args)
        x_test, y_test = joblib_to_x_and_y( os.path.join(experiment_root_dir, 'test_release.joblib'), names_of_feats_to_use , MASK_TO_BIN, args)
        training_main(
            model_type = model,
            samplewise_embedder = sample_wise_embedder,
            save_dir = output_dir, 
            pre_splitted=True,
            x_train=x_train, x_val=x_val, x_test=x_test, y_train=y_train, y_val=y_val, y_test=y_test, test_mode=test_mode, args=args
        )

    else:
        print('no train.joblib, val.joblib and test.joblib found, did you distribute and make the experiments correctly?')
        print('tried to search the following')
        print(train_dat_dir)
        print(val_dat_dir)
        print(test_dat_dir)
        print(whole_dat_dir)


def training_main(model_type, samplewise_embedder, save_dir, pre_splitted:bool, X=None, y=None, x_train=None, x_val=None, x_test=None, y_train=None, y_val=None, y_test=None, test_mode=False, args=None):

    print('--- training main starting')
    import time
    start_time = time.time()

    print('skip if done? ', args.skip_if_done)
    if  os.path.exists(os.path.join(save_dir, 'model.joblib')) and args.skip_if_done=='yes':
        print('looks you already have a model in there, skipping')
        exit(0)

    if pre_splitted is True:
        for _ in [x_train, x_val, x_test, y_train, y_val, y_test]:
            if _ is None:
                print('you have to pass in all trainx, trainy, valx, valy, testx, testy for pre_splitted=True')
                raise TypeError
        assert set(y_train) == set(y_val), 'inconsistant label classes in ytrain and yval'
        _ = preprocessing.LabelEncoder().fit(y_train)
        mapping = dict(zip(_.classes_, _.transform(_.classes_)))
        label_embedder = ReactLabelEmbedder(mapping, _.classes_)

    else:
        print('please split the experiment before calling training_main')
        exit(0)

    print(f'\nno. train: {len(y_train):,}')
    print(f'no. val: {len(y_val):,}')
    print(f'no. test: {len(y_test):,}')
    print(f'no. features: {len(x_train[0]):,}')

    # shuffle the data
    x_train, y_train = sk_shuffle(x_train, y_train)
    x_val, y_val = sk_shuffle(x_val, y_val)

    # not shuffling test data

    if args.train_frac < 1.0:
        assert args.train_frac > 0.0
        n_train = int(len(x_train) * args.train_frac)
        x_train = x_train[:n_train]
        y_train = y_train[:n_train]

    print(x_train[0])

    print('\nyour label mapping is')
    print(label_embedder.get_label_mapping())
    print('\nyour inverse label mapping is')
    print(label_embedder.get_inverse_label_mapping())

    train_loader = ReactDataLoader(x_train, y_train)
    val_loader = ReactDataLoader(x_val, y_val)
    test_loader = ReactDataLoader(x_test, y_test)
    print('--- before oversample')
    stats_before_oversample = 'train: '+str(train_loader)+'\n'+'va;: '+str(val_loader)+'\n'+'test: '+str(test_loader)
    print(stats_before_oversample)
    print('not oversampling test_loader')

    if args.oversample_train == 'yes':
        train_loader.oversample() 
        val_loader.oversample()
        # not over sampling test loader
        print('--- after oversample')
        stats_after_oversample = 'train: '+str(train_loader)+'\n'+'val: '+str(val_loader)
        print(stats_after_oversample)

        print("-over sample is done, took-- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()

        vim_write_zz(
            os.path.join(save_dir, 'oversample_statistics.txt'),
            string='pre-oversample\n' + stats_before_oversample + 'post-oversample\n' + stats_after_oversample
        )

        class_weight = None

    else:
        val_loader.oversample()
        class_weight = 'balanced'

    mkdir_if_dne(save_dir)
    clear_dir(save_dir)

    import multiprocessing
    ncpu = multiprocessing.cpu_count()
    print('btw you have ', ncpu, ' cpus avail')

    ss = StandardScaler()
    if model_type in ('lr', 'LR'):
        clf_temp = LogisticRegression(solver=args.solver, penalty=args.penalty, random_state=0,
                                      n_jobs=args.model_n_jobs, class_weight=class_weight)
        param_grid = {'logisticregression__C': [1e-1, 1e0]}
    elif model_type in ('lgb', 'LGB'):
        clf_temp = LGBMClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth,
                                  num_leaves=args.num_leaves, random_state=0, n_jobs=args.model_n_jobs,
                                  class_weight=class_weight)
        param_grid = {'lgbmclassifier__n_estimators': [50, 100],
                      'lgbmclassifier__max_depth': [3, 5],
                      'lgbmclassifier__num_leaves': [2, 15],
                      'lgbmclassifier__boosting_type': ['gbdt']}
    elif model_type in ('rf','RF'):
        clf_temp = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth,
                                  random_state=0, n_jobs=args.model_n_jobs,class_weight=class_weight)
        param_grid = {'randomforestclassifier__n_estimators': [50, 100],
                      'randomforestclassifier__max_depth': [3, 5],
                      'randomforestclassifier__min_samples_leaf': [2, 4]}
    elif model_type in ('dt','DT'):
        clf_temp = DecisionTreeClassifier(random_state=0, class_weight=class_weight)
        param_grid = {'decisiontreeclassifier__max_depth': [3, 5, None],
                      'decisiontreeclassifier__min_samples_leaf': [1, 2, 4, 10]}
    pipeline = make_pipeline(ss, clf_temp)
    if args.tune == 'yes':
        clf = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3, verbose=3, n_jobs=args.cv_n_jobs)
    else:
        clf = clf_temp if args.model == 'lgb' else pipeline

    if test_mode:
        clf = clf_temp

    model = ReactClassicalModel(
        classifier=clf,
        sample_wise_embedder=samplewise_embedder,
        label_embedder=label_embedder
    )

    trainer = ReactClassicalModelTrainer(
        model=model,
        serialization_dir=save_dir,
        data_loader=train_loader,
        validation_data_loader=val_loader,
        test_data_loader=test_loader

    )

    print("Starting training")
    trainer.train()
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Finished training")

    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f'max_rss: {max_rss:,}b')


if __name__ == '__main__':
    import argparse
    
    # must have cms
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', type=str, help='dir for that experiment')
    parser.add_argument('--feature_setting', type=str, default='all',help='set of features to use, must be one of the ones one spreadsheet')
    parser.add_argument('--model', type=str, default='lgb', help='specify your linear model: lr or lgb or rf.')
    parser.add_argument('--skip_if_done', type=str, default='yes', help='skip if an exp is already runned')
    parser.add_argument('--test', type=str, default='no', help='if yes only run pretty quick')
    parser.add_argument('--model_n_jobs', type=int, default=1, help='no. jobs to run in parallel for the model.')
    parser.add_argument('--cv_n_jobs', type=int, default=1, help='no. jobs to run in parallel for gridsearch.')
    parser.add_argument('--solver', type=str, default='lbfgs', help='LR solver.')
    parser.add_argument('--penalty', type=str, default='l2', help='LR penalty: l1 or l2.')
    parser.add_argument('--train_frac', type=float, default=1.0, help='fraction of train data to train with.')
    parser.add_argument('--oversample_train', type=str, default='no', help='yes for oversampling, no otherwise.')
    parser.add_argument('--n_estimators', type=int, default=100, help='no. boosting rounds for lgb.')
    parser.add_argument('--max_depth', type=int, default=5, help='max. depth for each tree.')
    parser.add_argument('--num_leaves', type=int, default=32, help='no. leaves per tree.')
    parser.add_argument('--tune', type=str, default='yes', help='if yes, then tune, otherwise no.')
    parser.add_argument('--detection_bert_root', type=str, default=None, help='if valid, use bert in that root to re-encode tp_bert')
    args = parser.parse_args()

    print('starting ', args.experiment_dir)

    if args.model == None:
        print('--- no model arch given, defaulting to lr')
        args.model = 'lr' #defaulting to logistic regression

    out_dir = args.experiment_dir
    print('--- searching exp data in ', out_dir)
    assert os.path.exists(os.path.join( out_dir, 'setting.json')), 'cannot find setting.json in '+out_dir+', did you make the experiment?'
    assert os.path.exists(os.path.join( out_dir, 'train.csv')), 'cannot find train.csv in '+out_dir+', did you make the experiment?'
    assert os.path.exists(os.path.join(out_dir, 'val.csv')), 'cannot find val.csv in '+out_dir+', did you make the experiment?'
    assert os.path.exists(os.path.join(out_dir, 'test_release.csv')), 'cannot find test_release.csv in '+out_dir+', did you make the experiment?'
    exp_args = load_json( os.path.join( out_dir, 'setting.json') )
    exp_args['skip_if_done'] = args.skip_if_done 
    exp_args['test'] = args.test 
    exp_args['model'] = args.model 
    exp_args['feature_setting'] = args.feature_setting
    exp_args['model_n_jobs'] = args.model_n_jobs
    exp_args['cv_n_jobs'] = args.cv_n_jobs
    exp_args['solver'] = args.solver
    exp_args['penalty'] = args.penalty
    exp_args['train_frac'] = args.train_frac
    exp_args['oversample_train'] = args.oversample_train
    exp_args['n_estimators'] = args.n_estimators
    exp_args['max_depth'] = args.max_depth
    exp_args['num_leaves'] = args.num_leaves
    exp_args['tune'] = args.tune
    exp_args['detection_bert_root'] = args.detection_bert_root
    exp_args = argparse.Namespace(**exp_args)

    TEST_MODE = False 
    if args.test == 'yes':
        TEST_MODE = True
    
    run_experiment(
        experiment_root_dir=out_dir,
        feature_setting=args.feature_setting,
        setting=exp_args.experiment_setting,
        model=args.model,
        test_mode=TEST_MODE,
        args=exp_args
        )