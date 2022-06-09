
## tcab_benchmark

This repository contains code for running classification experiments on adversarial attacks.

<img src="images/TCAB_Architecture.png" style="zoom:60%;" />


## Requirements

First install dependencies, eg:

```
conda create -n react-detection python=3.8
source activate react-detection
pip install -r requirements.txt
```

Then add reactdetect folder to you environment. E.g. if you are using conda

```
conda develop <absolute path to reactdetect>
```

Then, plug into ```react-detect/attacks_dataset``` folder an ```attacks``` folder in convention of the [dataset generation repo](https://github.com/a1noack/react/tree/master/classification), our use pre-generated attack instances [here](https://drive.google.com/drive/u/0/folders/11LRMS9ma1FKEnlYBbv5KtiiaNvLTTY51). (not required for quickstart)

## Quick-start and Examples

The ```reactdetect``` folder acts as a small library that works as the building block for other modules in the repo. For a quick glance of how to train a sklearn or pytoch type model, see various examples and annotated jupyter notebook in ```./reactdetect/examples```.

## Running Experiments at Scale

As the feature extraction process could be reused for various experiments, the repo contains another workflow for storing pre-computed features and running experiments at scale. 

* First, run ```encode_main.py``` to generate sentence level feature representations at ```./reprs/samplewise```. (Similarly ```encode_main_tw.py``` generates token level features at ```./reprs/tokenwise```)

* Then run ```distribute_experiments.py``` with proper arguments, which will create ```train.csv``` and ```test.csv``` under ```./detection-experiments``` folder.

* Then run ```make_experiments.py```, which takes in any directory that contains train and test csvs and make them into joblib files using cached representations in ```./reprs``` folder. You can disable distribution of token-wise features via its commandline args.

* Finally, run ```detection_sw.py``` that takes in any directory that contains pre-made train and test joblib files, which trains a LR/LGB/RF model per its input arguments; and logs models, outputs and metrics in a unique subdirectory.



## Reactdetect

./reactdetect contains the building blocks for the experiments, it is formulated to work as a small library when added to PYTHONPATH. 

```
reactdetect/
├── aggregation/        # modules that does "string to id/vector" handling, eg token/feature embedder, etc
├── allennlp_wrappers/  # relevant classes that overrides original allennlp modules, e.g. trainer that takes in python list
├── data/               # dataset readers, data loaders, etc. The classes stays as close of allennlp ones as possible
├── examples/           # examples
├── feature_extraction/ # feature extractor class and feature extaction functions
├── featu..._tokenwise/ # feature extractor class and feature extaction functions, at token level
├── utils/              # helpful file io, pandas operations and magic vars, etc.
└── models/             # base model classes, the neural base model stays as close of original allennlp one as possible
```



