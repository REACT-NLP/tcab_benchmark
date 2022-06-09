"""
utils, written by Jonathan
"""

"""
Utility methods to make life easier.
"""
import os
import pandas as pd
import sys
import time
import shutil
import logging
import itertools

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
from filelock import FileLock
from filelock import Timeout
from mpl_toolkits.axes_grid1 import make_axes_locatable

from models import BERTClassifier

def convert_nested_list_to_df(df_list):
    """
    Converts a list of pd.DataFrame objects into one pd.DataFrame object.
    """
    return pd.concat(df_list)

def get_logger(filename=''):
    """
    Return a logger object to easily save textual output.
    """

    logger = logging.getLogger()
    logger.handlers = []  # clear previous handlers
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    log_handler = logging.FileHandler(filename, mode='w')
    formatter = logging.Formatter('%(message)s')

    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(log_handler)

    return logger


def remove_logger(logger):
    """
    Remove handlers from logger.
    """
    logger.handlers = []


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


def get_model(model_name, max_seq_len=250, num_labels=2, tf_vectorizer=None, tfidf_vectorizer=None):
    """
    Return a new instance of the text classification model.
    """
    if model_name == 'bert':
        model = BERTClassifier(max_seq_len=max_seq_len, num_labels=num_labels)

    else:
        raise ValueError('Unknown model {}!'.format(model_name))

    return model


def get_loss_fn(loss_fn_name, weight=None):
    """
    Choose loss function.
    """
    if loss_fn_name == 'crossentropy':
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)

    else:
        raise ValueError('Unknown loss_fn {}'.format(loss_fn_name))

    return loss_fn


def get_optimizer(optimizer_name, lr, model, weight_decay=0.0):
    """
    Choose optimizer.
    """
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    elif optimizer_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    else:
        raise ValueError('Unknown optimizer {}'.format(optimizer_name))

    return optimizer





def batch_encode(tokenizer, text_list):
    """
    Encode list of text into lists of tokens.
    """
    if hasattr(tokenizer, "batch_encode"):
        result = tokenizer.batch_encode(text_list)
    else:
        result = [tokenizer.encode(text_input) for text_input in text_list]
    return result


def generate_multiclass_roc_curves(y_true, y_score, class_names=None):
    """
    Returns a dictionary of One vs. Rest ROC curves. Also includes
    a macro ROC curve.
    Input
    y_true: 1d arry of class label integers
    y_score: 2d array of shape=(no. samples, no. classes)
    label_map: 1d list of class names.
    """

    # binarize the output
    n_classes = y_score.shape[1]
    y_true = label_binarize(y_true, classes=list(range(n_classes)))

    # create class names if None
    if class_names is None:
        class_names = ['class_{}'.format(i) for i in range(n_classes)]

    # compute ROC curve and ROC area for each class
    roc_curves = {}
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_curves[class_names[i]] = (fpr, tpr, None)

    # first aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr for k, (fpr, tpr, _) in roc_curves.items()]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for k, (fpr, tpr, _) in roc_curves.items():
        mean_tpr += np.interp(all_fpr, fpr, tpr)

    # finally average it
    mean_tpr /= n_classes
    roc_curves['Macro Average'] = (all_fpr, mean_tpr, None)

    return roc_curves


def plot_roc_curves(curves, ax=None, zoom=False, width=18, legend_fontsize=7.5):
    """
    Plot ROC curve.
    """
    golden_ratio = 1.61803

    if ax is None:
        fig, ax = plt.figure(figsize=(width, width / golden_ratio))

    ax.set_title('ROC curves')

    ax.set_ylabel("True Positive Rate")
    ax.set_ylim([-0.05, 1.05])
    ax.set_yticks(np.arange(0, 1, 0.1), minor=True)

    ax.set_xlabel("False Positive Rate")
    ax.set_xticks(np.arange(0, 1, 0.1), minor=True)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.01))

    if zoom:
        ax.set_xlim([0.0, 0.01])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.001))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.0001))

    ax.plot([0, 1], [0, 1], "k:", label="Random")
    for name, (fpr, tpr, thresholds) in curves.items():
        auc_score = auc(fpr, tpr)
        ax.plot(fpr, tpr, label='{}: {:.3f}'.format(name, auc_score))

    ax.legend(loc="lower right", fontsize=legend_fontsize)
    ax.grid(b=True, which='major')
    ax.grid(b=True, which='minor', linewidth=0.1)

    return ax


def plot_confusion_matrix(cm,
                          target_names,
                          cmap=None,
                          normalize=True,
                          ax=None):
    """
    given a sklearn confusion matrix (cm), make a nice plot
    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix
    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']
    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues
    normalize:    If False, plot the raw numbers
                  If True, plot the proportions
    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph
    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if ax is None:
        fig, ax = plt.figure(figsize=(8, 6))

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, orientation='vertical')

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(target_names, rotation=45, ha='right')
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            ax.text(j, i, '{:.1f}'.format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        else:
            ax.text(j, i, '{:.1f}'.format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')