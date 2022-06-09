"""
Driver for invoking train.main with additinal args
"""
import os
import argparse
from easydict import EasyDict as edict
from train import main as training_main
from data_partition import main as data_partition_main

def get_args_for_bert_training(dataset_parent_dir, dataset_folder_name, output_dir):
    """
    Create a instance of args with several pre-filled fields.
    This generates args for train.main.
    """
    args_for_bert = {
        'out_dir': output_dir,
        'data_dir': dataset_parent_dir,
        'dataset': dataset_folder_name,
        'model': 'bert',
        'rs': 1,
        'loss_fn': 'crossentropy',
        'optimizer': 'adam',
        'max_seq_len': 250,
        'lr': 1e-06,
        'batch_size': 32,
        'epochs': 10,
        'weight_decay': 0.0,
        'max_norm': 1.0,
        'nrows': 1000000000
    }
    return edict(args_for_bert)

def main(args):
    """
    create subfolder, generate csv for bert training
    """
    print('generating dataset')
    data_partition_main(args.experiment_dir)
    bert_args = get_args_for_bert_training(
        dataset_parent_dir = args.experiment_dir, 
        dataset_folder_name = 'BERT_DETECTION', 
        output_dir = os.path.join(args.experiment_dir, 'BERT_DETECTION')
        )
    print('starting training main')
    training_main(bert_args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', type=str, help='dir for that experiment')
    args = parser.parse_args()
    main(args)

