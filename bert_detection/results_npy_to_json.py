import numpy as np
import json
import os

def main(args):
    results_numpy_dir = os.path.join(args.experiment_dir, 'results.npy')
    results_json_dir = os.path.join(args.experiment_dir, 'results.json')
    if os.path.exists(results_numpy_dir) and not os.path.exists(results_json_dir):
        results = np.load(results_numpy_dir, allow_pickle=True)
        print(results)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', type=str, help='dir for that experiment')
    args = parser.parse_args()
    main(args)