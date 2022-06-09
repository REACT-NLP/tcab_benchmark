  
"""
This module contains a wrapper with an allennlp trainer
    with necessary files overriden so that it works with
    react conventions
"""
from allennlp.training.trainer import GradientDescentTrainer, Trainer
import datetime
import logging
import math
import numpy
import os
import pandas as pd
import re
import time
import traceback
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, Iterable

from allennlp.common.util import int_to_device

import torch
import torch.distributed as dist
from torch.cuda import amp
import torch.optim.lr_scheduler
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_

from allennlp.common import Lazy, Registrable, Tqdm
from allennlp.common import util as common_util
from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.data import DataLoader, TensorDict
from allennlp.models.model import Model
from allennlp.training import util as training_util
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.metric_tracker import MetricTracker
from allennlp.training.momentum_schedulers import MomentumScheduler
from allennlp.training.moving_average import MovingAverage
from allennlp.training.optimizers import Optimizer
# from allennlp.training.tensorboard_writer import TensorBoardWriter
from itertools import islice
from allennlp.training.trainer import *
#zhouhanx: seems like older allenNLP version don't support the following.
#from allennlp.sanity_checks.normalization_bias_verification import NormalizationBiasVerification
#from allennlp.training.log_writer import LogWriter

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize

import joblib
import json
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from textwrap import wrap


from reactdetect.utils.file_io import mkfile_if_dne, mkdir_if_dne


def react_groups_of(dataloader, group_size: int) -> (Iterable, Iterable):
    """
    original allenNLP:
    Takes an iterable and batches the individual instances into lists of the
    specified size. The last list may be smaller if there are instances left over.
    zhouhanx:
    modified this to work on (list, list), as in react we will likely deal with
    (list[texts],list[labels])
    this is no longer "really" lazy but for some strings this is probably fine.
    """
    index_iterator = iter(list([i for i in range(len(dataloader[0]))]))
    while True:
        s = list(islice(index_iterator, group_size))
        if len(s) > 0:
            pair = ([dataloader[0][i] for i in s], [dataloader[1][i] for i in s])
            yield pair
        else:
            break

def lazy_groups_of(iterable, group_size) :
    """
    Takes an iterable and batches the individual instances into lists of the
    specified size. The last list may be smaller if there are instances left over.
    """
    iterator = iter(iterable)
    while True:
        s = list(islice(iterator, group_size))
        #print(s)
        if len(s) > 0:
            yield s
        else:
            break

class ReactGradientDescentTrainer(GradientDescentTrainer):
    def __init__(
        self,
        model: Model,
        optimizer: torch.optim.Optimizer,
        data_loader: DataLoader,
        patience: Optional[int] = None,
        validation_metric: Union[str, List[str]] = "-loss",
        validation_data_loader: DataLoader = None,
        num_epochs: int = 20,
        serialization_dir: Optional[str] = None,
        checkpointer: Checkpointer = None,
        cuda_device: Optional[Union[int, torch.device]] = None,
        grad_norm: Optional[float] = None,
        grad_clipping: Optional[float] = None,
        learning_rate_scheduler: Optional[LearningRateScheduler] = None,
        momentum_scheduler: Optional[MomentumScheduler] = None,
        moving_average: Optional[MovingAverage] = None,
        callbacks: List[TrainerCallback] = None,
        distributed: bool = False,
        local_rank: int = 0,
        world_size: int = 1,
        num_gradient_accumulation_steps: int = 1,
        use_amp: bool = False,
    ) -> None:

        #(zhouhanx): newer allennlp's trainer except a model that's ***already***
        #    on the right device. Doing it before calling super().__init__() in case 
        #    there are some internal dependencies of objects that have dynamical device
        #    dependency based on model.device()

        if cuda_device is not None:
            model.to(cuda_device)
        

        super().__init__(
            model=model,
            optimizer=optimizer,
            data_loader=data_loader,
            patience=patience,
            validation_metric=validation_metric,
            validation_data_loader=validation_data_loader,
            num_epochs=num_epochs,
            serialization_dir=serialization_dir,
            checkpointer=checkpointer,
            cuda_device=cuda_device,
            grad_norm=grad_norm,
            grad_clipping=grad_clipping,
            learning_rate_scheduler=learning_rate_scheduler,
            momentum_scheduler=momentum_scheduler,
            moving_average=moving_average,
            callbacks=callbacks,
            distributed=distributed,
            local_rank=local_rank,
            world_size=world_size,
            num_gradient_accumulation_steps=num_gradient_accumulation_steps,
            use_amp=use_amp,
        )

    def batch_outputs(self, batch: (List[str],List[str]), for_training: bool) -> Dict[str, torch.Tensor]:
        """
        Does a forward pass on the given batch and returns the output dictionary that the model
        returns, after adding any specified regularization penalty to the loss (if training).
        """
        #print('--batch: ',batch)
        try:
            assert(type(batch[0])==type(batch[1])==list )
        except:
            print('batch_outputs get unexpected data format: ', len(batch), type(batch))
            print('batch_outputs get unexpected data format: ', len(batch[0]), type(batch[0]))
            print('batch_outputs get unexpected data format: ', len(batch[1]), type(batch[1]))
            print('batch_outputs get unexpected data format: ', batch)
            raise
        texts, labels = batch[0],batch[1]
        output_dict = self._pytorch_model(texts, labels)

        if for_training:
            try:
                assert "loss" in output_dict
                regularization_penalty = self.model.get_regularization_penalty()

                if regularization_penalty is not None:
                    output_dict["reg_loss"] = regularization_penalty
                    output_dict["loss"] += regularization_penalty

            except AssertionError:
                if for_training:
                    raise RuntimeError(
                        "The model you are trying to optimize does not contain a"
                        " 'loss' key in the output of model.forward(inputs)."
                    )

        return output_dict

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Trains one epoch and returns metrics.
        """
        logger.info("Epoch %d/%d", epoch, self._num_epochs - 1)
        cpu_memory_usage = []
        for worker, memory in common_util.peak_cpu_memory().items():
            cpu_memory_usage.append((worker, memory))
            logger.info(f"Worker {worker} memory usage: {common_util.format_size(memory)}")
        gpu_memory_usage = []
        for gpu, memory in common_util.peak_gpu_memory().items():
            gpu_memory_usage.append((gpu, memory))
            logger.info(f"GPU {gpu} memory usage: {common_util.format_size(memory)}")

        regularization_penalty = self.model.get_regularization_penalty()

        train_loss = 0.0
        batch_loss = 0.0
        train_reg_loss = None if regularization_penalty is None else 0.0
        batch_reg_loss = None if regularization_penalty is None else 0.0

        # Set the model to "train" mode.
        self._pytorch_model.train()

        # Get tqdm for the training batches
        batch_generator = self.data_loader
        batch_group_generator = react_groups_of(
            batch_generator, self._num_gradient_accumulation_steps
        )

        logger.info("Training")

        num_training_batches: Union[int, float]
        try:
            len_data_loader = len(self.data_loader)
            num_training_batches = math.ceil(
                len_data_loader / self._num_gradient_accumulation_steps
            )
        except TypeError:
            num_training_batches = float("inf")

        # Having multiple tqdm bars in case of distributed training will be a mess. Hence only the primary's
        # progress is shown
        if self._primary:
            batch_group_generator_tqdm = Tqdm.tqdm(
                batch_group_generator, total=num_training_batches
            )
        else:
            batch_group_generator_tqdm = batch_group_generator

        self._last_log = time.time()

        batches_this_epoch = 0
        if self._batch_num_total is None:
            self._batch_num_total = 0

        done_early = False

        for batch_group in batch_group_generator_tqdm:
            if self._distributed:
                # Check whether the other workers have stopped already (due to differing amounts of
                # data in each). If so, we can't proceed because we would hang when we hit the
                # barrier implicit in Model.forward. We use a IntTensor instead a BoolTensor
                # here because NCCL process groups apparently don't support BoolTensor.
                done = torch.tensor(0, device=self.cuda_device)
                torch.distributed.all_reduce(done, torch.distributed.ReduceOp.SUM)
                if done.item() > 0:
                    done_early = True
                    logger.warning(
                        f"Worker {torch.distributed.get_rank()} finishing training early! "
                        "This implies that there is an imbalance in your training "
                        "data across the workers and that some amount of it will be "
                        "ignored. A small amount of this is fine, but a major imbalance "
                        "should be avoided. Note: This warning will appear unless your "
                        "data is perfectly balanced."
                    )
                    break

            batches_this_epoch += 1
            self._batch_num_total += 1
            batch_num_total = self._batch_num_total

            # Zero gradients.
            # NOTE: this is actually more efficient than calling `self.optimizer.zero_grad()`
            # because it avoids a read op when the gradients are first updated below.
            for param_group in self.optimizer.param_groups:
                for p in param_group["params"]:
                    p.grad = None

            batch_loss = 0.0
            batch_group_outputs = []
            #print('---batch group is ',batch_group)
            
            with amp.autocast(self._use_amp):
                batch_outputs = self.batch_outputs(batch_group, for_training=True)
                batch_group_outputs.append(batch_outputs)
                loss = batch_outputs["loss"]
                reg_loss = batch_outputs.get("reg_loss")
                if torch.isnan(loss):
                    raise ValueError("nan loss encountered")
                loss = loss / len(batch_group)

                batch_loss += loss.item()
                if reg_loss is not None:
                    reg_loss = reg_loss / len(batch_group)
                    batch_reg_loss = reg_loss.item()
                    train_reg_loss += batch_reg_loss  # type: ignore

            if self._scaler is not None:
                self._scaler.scale(loss).backward()
            else:
                loss.backward()

            train_loss += batch_loss

            batch_grad_norm = self.rescale_gradients()

            # This does nothing if batch_num_total is None or you are using a
            # scheduler which doesn't update per batch.
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step_batch(batch_num_total)
            if self._momentum_scheduler:
                self._momentum_scheduler.step_batch(batch_num_total)

            if self._scaler is not None:
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                self.optimizer.step()

            # Update moving averages
            if self._moving_average is not None:
                self._moving_average.apply(batch_num_total)

            # Update the description with the latest metrics
            metrics = training_util.get_metrics(
                self.model,
                train_loss,
                train_reg_loss,
                batch_loss,
                batch_reg_loss,
                batches_this_epoch,
                world_size=self._world_size,
                cuda_device=self.cuda_device,
            )

            if self._primary:
                # Updating tqdm only for the primary as the trainers wouldn't have one
                description = training_util.description_from_metrics(metrics)
                batch_group_generator_tqdm.set_description(description, refresh=False)

                if self._checkpointer is not None:
                    self._checkpointer.maybe_save_checkpoint(self, epoch, batches_this_epoch)

            for callback in self._callbacks:
                callback.on_batch(
                    self,
                    batch_group,
                    batch_group_outputs,
                    metrics,
                    epoch,
                    batches_this_epoch,
                    is_training=True,
                    is_primary=self._primary,
                    batch_grad_norm=batch_grad_norm,
                )

        if self._distributed and not done_early:
            logger.warning(
                f"Worker {torch.distributed.get_rank()} completed its entire epoch (training)."
            )
            # Indicate that we're done so that any workers that have remaining data stop the epoch early.
            done = torch.tensor(1, device=self.cuda_device)
            torch.distributed.all_reduce(done, torch.distributed.ReduceOp.SUM)
            assert done.item()

        # Let all workers finish their epoch before computing
        # the final statistics for the epoch.
        if self._distributed:
            dist.barrier()

        metrics = training_util.get_metrics(
            self.model,
            train_loss,
            train_reg_loss,
            batch_loss=None,
            batch_reg_loss=None,
            num_batches=batches_this_epoch,
            reset=True,
            world_size=self._world_size,
            cuda_device=self.cuda_device,
        )

        for (worker, memory) in cpu_memory_usage:
            metrics["worker_" + str(worker) + "_memory_MB"] = memory / (1024 * 1024)
        for (gpu_num, memory) in gpu_memory_usage:
            metrics["gpu_" + str(gpu_num) + "_memory_MB"] = memory / (1024 * 1024)
        return metrics

    def _validation_loss(self, epoch: int) -> Tuple[float, Optional[float], int]:
        """
        Computes the validation loss. Returns it and the number of batches.
        """
        logger.info("Validating")

        self._pytorch_model.eval()

        # Replace parameter values with the shadow values from the moving averages.
        if self._moving_average is not None:
            self._moving_average.assign_average_value()

        if self._validation_data_loader is not None:
            validation_data_loader = self._validation_data_loader
        else:
            raise ConfigurationError(
                "Validation results cannot be calculated without a validation_data_loader"
            )

        regularization_penalty = self.model.get_regularization_penalty()

        # Having multiple tqdm bars in case of distributed training will be a mess. Hence only the primary's
        # progress is shown
        if self._primary:
            val_generator_tqdm = Tqdm.tqdm(validation_data_loader)
        else:
            val_generator_tqdm = validation_data_loader

        batches_this_epoch = 0
        val_loss = 0.0
        val_batch_loss = 0.0
        val_reg_loss = None if regularization_penalty is None else 0.0
        val_batch_reg_loss = None if regularization_penalty is None else 0.0
        done_early = False
        
        #for batch in val_generator_tqdm:
        for i in range(1): #(zhouhanx)seems like there is no actual split, for validation the whole thing is fed over.
            if self._distributed:
                # Check whether the other workers have stopped already (due to differing amounts of
                # data in each). If so, we can't proceed because we would hang when we hit the
                # barrier implicit in Model.forward. We use a IntTensor instead a BoolTensor
                # here because NCCL process groups apparently don't support BoolTensor.
                done = torch.tensor(0, device=self.cuda_device)
                torch.distributed.all_reduce(done, torch.distributed.ReduceOp.SUM)
                if done.item() > 0:
                    done_early = True
                    logger.warning(
                        f"Worker {torch.distributed.get_rank()} finishing validation early! "
                        "This implies that there is an imbalance in your validation "
                        "data across the workers and that some amount of it will be "
                        "ignored. A small amount of this is fine, but a major imbalance "
                        "should be avoided. Note: This warning will appear unless your "
                        "data is perfectly balanced."
                    )
                    break

            with amp.autocast(self._use_amp):
                batch_outputs = self.batch_outputs( (validation_data_loader[0], validation_data_loader[1]), for_training=False)
                loss = batch_outputs.get("loss")
                reg_loss = batch_outputs.get("reg_loss")
                if loss is not None:
                    # You shouldn't necessarily have to compute a loss for validation, so we allow for
                    # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
                    # currently only used as the divisor for the loss function, so we can safely only
                    # count those batches for which we actually have a loss.  If this variable ever
                    # gets used for something else, we might need to change things around a bit.
                    batches_this_epoch += 1
                    val_batch_loss = loss.item()
                    val_loss += val_batch_loss
                    if reg_loss is not None:
                        val_batch_reg_loss = reg_loss.item()
                        val_reg_loss += val_batch_reg_loss  # type: ignore

            # Update the description with the latest metrics
            val_metrics = training_util.get_metrics(
                self.model,
                val_loss,
                val_reg_loss,
                val_batch_loss,
                val_batch_reg_loss,
                batches_this_epoch,
                world_size=self._world_size,
                cuda_device=self.cuda_device,
            )

            description = training_util.description_from_metrics(val_metrics)
            if self._primary:
                val_generator_tqdm.set_description(description, refresh=False)

            for callback in self._callbacks:
                callback.on_batch(
                    self,
                    [batch],
                    [batch_outputs],
                    val_metrics,
                    epoch,
                    batches_this_epoch,
                    is_training=False,
                    is_primary=self._primary,
                )

        if self._distributed and not done_early:
            logger.warning(
                f"Worker {torch.distributed.get_rank()} completed its entire epoch (validation)."
            )
            # Indicate that we're done so that any workers that have remaining data stop validation early.
            done = torch.tensor(1, device=self.cuda_device)
            torch.distributed.all_reduce(done, torch.distributed.ReduceOp.SUM)
            assert done.item()

        # Now restore the original parameter values.
        if self._moving_average is not None:
            self._moving_average.restore()

        return val_loss, val_reg_loss, batches_this_epoch


class ReactClassicalModelTrainer():
    def __init__(
        self,
        model,
        data_loader: DataLoader,
        validation_data_loader: DataLoader,
        test_data_loader: DataLoader,
        serialization_dir: Optional[str] = None,
        metrics: dict() = None,
    ) -> None:
        self.model = model 
        self.data_loader = data_loader 
        self.validation_data_loader = validation_data_loader
        self.test_data_loader = test_data_loader
        self.serialization_dir = serialization_dir
        self.metrics = metrics
    
    def train(self) -> None:
        """
        As the self.model.classifier could be several things, there might be more checkings or try-except needed, e.g. decide whether there is a LR model somewhere, etc
        """
        embedded_texts_train, embedded_labels_train = self.model.embed_texts_and_labels(self.data_loader.get_texts(), self.data_loader.get_labels())

        self.model.classifier.fit(embedded_texts_train, embedded_labels_train)
        y_pred_train = self.model.classifier.predict(embedded_texts_train)   
        acc_train = accuracy_score(y_pred_train, embedded_labels_train)
        print('training accuracy: ',acc_train)
        bacc_train = balanced_accuracy_score(y_pred_train, embedded_labels_train)
        print('training balanced accuracy: ',bacc_train)     

        embedded_texts_validation, embedded_labels_validation = self.model.embed_texts_and_labels(self.validation_data_loader.get_texts(), self.validation_data_loader.get_labels())
        y_pred_val = self.model.classifier.predict(embedded_texts_validation)    
        acc_validation = accuracy_score(y_pred_val, embedded_labels_validation)
        print('validation accuracy: ', acc_validation)
        bacc_validation = balanced_accuracy_score(y_pred_val, embedded_labels_validation)
        print('validation balanced accuracy: ',bacc_validation)

        embedded_texts_test, embedded_labels_test = self.model.embed_texts_and_labels(self.test_data_loader.get_texts(), self.test_data_loader.get_labels())
        y_pred_test = self.model.classifier.predict(embedded_texts_test)    
        acc_test = accuracy_score(y_pred_test, embedded_labels_test)
        print('test: ', acc_test)
        bacc_test = balanced_accuracy_score(y_pred_test, embedded_labels_test)
        print('test balanced accuracy: ',bacc_test)

        self._save_true_test_labels_for_leaderboard_release(y_pred_test, self.model.label_embedder.get_inverse_label_mapping())
    
        ## useful info for recreating roc
        data_dump_for_roc = {
            'y_pred_val':y_pred_val,
            'y_true_val':embedded_labels_validation,
            'y_proba_val':self.model.classifier.predict_proba(embedded_texts_validation),
            'x_val':embedded_texts_validation,
            'y_val':embedded_labels_validation,
            'y_pred_test':y_pred_test,
            'y_true_test':embedded_labels_test,
            'y_proba_test':self.model.classifier.predict_proba(embedded_texts_test),
            'x_test':embedded_texts_test,
            'y_test':embedded_labels_test,
            'label_mapping': self.model.label_embedder.get_inverse_label_mapping()
        }
        mkfile_if_dne(os.path.join(self.serialization_dir, 'data_dump_for_roc.joblib'))
        joblib.dump(data_dump_for_roc, os.path.join(self.serialization_dir, 'data_dump_for_roc.joblib') )

        conf_matrix_train = confusion_matrix(embedded_labels_train, y_pred_train).tolist()
        conf_matrix_val = confusion_matrix(embedded_labels_validation, y_pred_val).tolist()
        conf_matrix_test = confusion_matrix(embedded_labels_test, y_pred_test).tolist()

        self.metrics = {"training_accuracy": acc_train,"validation_accuracy": acc_validation, "test_accuracy": acc_test,
                        "training_balanced_accuracy":bacc_train, "validation_balanced_accuracy":bacc_validation,
                        "test_balanced_accuracy":bacc_test}
        self.metrics['confusion_matrix_train'] = conf_matrix_train
        self.metrics['confusion_matrix_validation'] = conf_matrix_val
        self.metrics['confusion_matrix_test'] = conf_matrix_test


        try:
            embedded_labels_train = embedded_labels_train.tolist() #in case it is a tensor 
        except:
            pass
        
        if len(set(embedded_labels_train))==2:
            
            train_roc_auc = roc_auc_score(embedded_labels_train, y_pred_train)
            validation_roc_auc = roc_auc_score(embedded_labels_validation, y_pred_val)
            test_roc_auc = roc_auc_score(embedded_labels_test, y_pred_test)
            self.metrics['train_roc_auc'] = train_roc_auc
            self.metrics['validation_roc_auc'] = validation_roc_auc
            self.metrics['test_roc_auc'] = test_roc_auc

        else:
            CLASS_NAMES = self.model.label_embedder.get_class_names()
            y_score_train = self.model.classifier.predict_proba(embedded_texts_train)
            train_roc_curves = self.model.generate_multiclass_roc_curves(embedded_labels_train, y_score_train, class_names=CLASS_NAMES)
            fig, ax = plt.subplots()
            self.model.plot_roc_curves(train_roc_curves)
            _ = os.path.join(self.serialization_dir, 'multiclass_roc_curves/train_roc_curve.png')
            mkfile_if_dne(_)
            plt.savefig(_, bbox_inches='tight')

            y_score_validation = self.model.classifier.predict_proba(embedded_texts_validation)
            validation_roc_curves = self.model.generate_multiclass_roc_curves(embedded_labels_validation, y_score_validation, class_names=CLASS_NAMES)
            fig, ax = plt.subplots()
            self.model.plot_roc_curves(validation_roc_curves)
            _ = os.path.join(self.serialization_dir, 'multiclass_roc_curves/validation_roc_curve.png')
            mkfile_if_dne(_)
            plt.savefig(_, bbox_inches='tight')

            y_score_test = self.model.classifier.predict_proba(embedded_texts_test)
            test_roc_curves = self.model.generate_multiclass_roc_curves(embedded_labels_test, y_score_test, class_names=CLASS_NAMES)
            fig, ax = plt.subplots()
            self.model.plot_roc_curves(test_roc_curves)
            _ = os.path.join(self.serialization_dir, 'multiclass_roc_curves/test_roc_curve.png')
            mkfile_if_dne(_)
            plt.savefig(_, bbox_inches='tight')
            pass
        try:
            self._extend_model_metrics()
        except Exception as e:
            print(e)
            print('--- from ReactTrainer: _extend_model_metrics failed, this is expected if ur model is not lr')
        self._save_model_and_metrics()

    def _extend_model_metrics(self) -> None:
    
        self.metrics['feature_names'] = list(self.model.sample_wise_embedder.get_feature_names()) # make sure these are lists since numpy ND arrays are not JSON serializable
        if isinstance(self.model.classifier, GridSearchCV):
            self.metrics['best_params'] = self.model.classifier.best_params_
            if isinstance(self.model.classifier.best_estimator_, Pipeline):
                self.metrics['coef'] = self.model.classifier.best_estimator_[-1].coef_.tolist()
                self.metrics['intercept'] = self.model.classifier.best_estimator_[-1].intercept_.tolist()
            else:
                self.metrics['coef'] = self.model.classifier.best_estimator_.coef_.tolist()
                self.metrics['intercept'] = self.model.classifier.best_estimator_.intercept_.tolist()
        else:
            self.metrics['coef'] = self.model.classifier.coef_.tolist() 
            self.metrics['intercept'] = self.model.classifier.intercept_.tolist() 
        self.metrics['label_mapping'] = self.model.label_embedder.get_inverse_label_mapping()
        feature_names = self.metrics['feature_names']
        self.metrics['important_features'] = self.model.extract_important_features(coefs=numpy.array(self.metrics['coef']), feature_names=feature_names)

    def _save_model_and_metrics(self) -> None:
        """
        Save model and a dictionary of metrics to self.serialization_dir
        """
        print('saving metrics to ', self.serialization_dir)
        mkdir_if_dne(self.serialization_dir)
        with open(os.path.join(self.serialization_dir,'metrics.json'),'w') as ofp:
            try:
                json.dump(self.metrics, ofp)
            except Exception as e:
                print('oops, saving metrics failed, reason is')
                print(e)
        try:
            joblib.dump(self.model.classifier, os.path.join(self.serialization_dir,'model.joblib') )
        except:
            print('oops, saving model failed, reasons is')
            print(e)

    def _save_true_test_labels_for_leaderboard_release(self, y_test_pred, inverse_label_mapping) -> None:
        y_test_labels = []

        for pred in y_test_pred:
            y_test_labels.append(inverse_label_mapping[pred])

        df_test_labels = pd.DataFrame(y_test_labels)
        df_test_labels.to_csv(os.path.join(self.serialization_dir, 'test_pred_labels.csv'), index=True, header=False)

if __name__ == '__main__':
    print('testing react_groups_of batching function')
    for g in react_groups_of((['hel', 'fds', 'fdsfdfdsf','aaaa','fdsfdfii'], [1,1,0,1,0]), 2):
        print(g)
    print('done')

    for g in lazy_groups_of([1,2,3,4,5],2):
        print(g)


 