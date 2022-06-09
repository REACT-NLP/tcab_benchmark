
from re import L
import joblib
import os
import pandas as pd

from reactdetect.utils.file_io import grab_joblibs
from reactdetect.utils.file_io import load_json

from reactdetect.utils.hashing import get_pk_tuple_old, get_pk_tuple

from reactdetect.utils.magic_vars import SUPPORTED_TARGET_MODELS
from reactdetect.utils.magic_vars import SUPPORTED_TARGET_MODEL_DATASETS

from reactdetect.utils.pandas_ops import no_duplicate_index

def check_joblib_dict(samples_dict):
    """
    Checks to make sure each example has the correct
        number of fields in the primary key; if not,
        this method attempts to fill in the missing values.

    Input
        samples_dict: dict of extracted features
            format: key - sample index, value - dict.

    Return
        samples_dict, with updated primary keys and unique IDs.
    """
    # actual primary key as len=7 pk in job has len=6. we take care of this discrepancy in fn df_to_instance_subset
    for key, d in samples_dict.items():
        assert len(d['primary_key']) == 6
    return samples_dict

def load_known_instances(rootdir_with_joblib_file, target_model, target_model_dataset, lazy_loading):
    assert target_model in SUPPORTED_TARGET_MODELS
    assert target_model_dataset in SUPPORTED_TARGET_MODEL_DATASETS
    
    print('\n--- loading known instances from ', rootdir_with_joblib_file)
    print(f'--- target_model: {target_model}, target_model_dataset: {target_model_dataset}')
    
    all_jblb = grab_joblibs(rootdir_with_joblib_file)
    
    if lazy_loading:
        all_jblb = [jb for jb in all_jblb if target_model + '_' in jb]
        all_jblb = [jb for jb in all_jblb if target_model_dataset + '_' in jb]
        
    print(f'--- No. joblib files of varied size to read: {len(all_jblb):,}')
    
    known_samples = {}
    
    for i, jblb in enumerate(all_jblb):
        print(f'[{i}] {jblb}')  # to avoid name collision
        
        holder = check_joblib_dict(joblib.load(jblb))

        for idx in holder.keys():
            instance = holder[idx]
            pk_old = instance['primary_key']
            pk_old = tuple(pk_old)
            assert isinstance(pk_old, tuple)
            known_samples[pk_old] = instance

    print(f'\ndone, no. unique instances with extracted features: {len(known_samples):,}')
    return known_samples

def df_to_instance_subset(df, known_samples):
    assert no_duplicate_index(df)

    out = {}
    no_repr_count = 0

    for idx in df.index:
        pk_old = get_pk_tuple_old(df, idx)
        pk = get_pk_tuple(df, idx)
        pk_old = tuple(pk_old)
        pk = tuple(pk)

        if pk_old in known_samples:
            out[pk] = known_samples[pk_old]
            del known_samples[pk_old]

            out[pk]['attack_name'] = pk[0]  # ok so this is because react convention, pk[0] is attack_name
            out[pk]['binary_label'] = 'clean' if pk[0] == 'clean' else 'perturbed'

            try:  # just in case we need the actual text
                out[pk]['perturbed_text'] = df.at[idx, 'perturbed_text']
                out[pk]['original_text'] = df.at[idx, 'original_text']

            except:
                pass

        else:
            no_repr_count += 1

    print(f'    cannot find repr. for {no_repr_count:,} / {len(df):,} instances')

    return out


def main(args, out_dir):
    print('making exp into... ', out_dir)

    lazy_loading = True
    known_instances = load_known_instances('./reprs/samplewise',
                                            target_model=args.target_model,
                                            target_model_dataset=args.target_model_dataset,
                                            lazy_loading=lazy_loading)

    print('\n--- creating train.joblib')
    train_df = pd.read_csv(os.path.join(out_dir, 'train.csv'))
    train_data = df_to_instance_subset(train_df, known_instances)
    joblib.dump(train_data, os.path.join(out_dir, 'train.joblib'))

    print('\n--- creating val.joblib')
    val_df = pd.read_csv(os.path.join(out_dir, 'val.csv'))
    val_data = df_to_instance_subset(val_df, known_instances)
    joblib.dump(val_data, os.path.join(out_dir, 'val.joblib'))

    print('--- creating test_release.joblib')
    test_df_release = pd.read_csv(os.path.join(out_dir, 'test_release.csv'))
    test_release_data = df_to_instance_subset(test_df_release, known_instances)
    joblib.dump(test_release_data, os.path.join(out_dir, 'test_release.joblib'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', type=str, help='directory off a distributed exp to be made')
    args = parser.parse_args()

    out_dir = args.experiment_dir

    assert os.path.exists(os.path.join(out_dir, 'setting.json')), 'cannot find setting.json in ' + out_dir
    assert os.path.exists(os.path.join(out_dir, 'train.csv')), 'cannot find train.csv in ' + out_dir
    assert os.path.exists(os.path.join(out_dir, 'val.csv')), 'cannot find train.csv in ' + out_dir
    assert os.path.exists(os.path.join(out_dir, 'test_release.csv')), 'cannot find test.csv in ' + out_dir

    exp_args = load_json(os.path.join(out_dir, 'setting.json'))
    exp_args = argparse.Namespace(**exp_args)

    main(exp_args, out_dir)