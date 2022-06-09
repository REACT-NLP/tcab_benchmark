from typing import Dict, List, Set, Type, Optional, Union, Iterable, Tuple
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy 
from reactdetect.feature_extraction.feature_extractor import *
from reactdetect.aggregation.feature_embedders import ReactTokenWiseFeatureEmbedder, ReactSampleWiseFeatureEmbedder
from reactdetect.aggregation.label_embedders import ReactLabelEmbedder
from allennlp.models import Model
from allennlp.modules import Seq2VecEncoder
from allennlp.modules.seq2vec_encoders import CnnEncoder,BagOfEmbeddingsEncoder
from allennlp.training.metrics import CategoricalAccuracy

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc



class ReactNeuralModel(Model):
    def __init__(self):
        super().__init__(vocab=None) #(zhouhanx): we'll never use vocab
        self.accuracy = CategoricalAccuracy()

    def forward(self, text: list, labels: list) -> Dict[str, torch.Tensor]:
        
        raise NotImplementedError

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        
        return {"accuracy": self.accuracy.get_metric(reset)}
        

    def forward_on_datalists(self, samples, labels) -> List[Dict[str, numpy.ndarray]]:
        
        outputs = self(samples, labels)
        outputs = self.make_output_human_readable(outputs)
        

        instance_separated_output: List[Dict[str, numpy.ndarray]] = [
            {} for _ in samples
        ]
        for name, output in list(outputs.items()):
            if isinstance(output, torch.Tensor):
                # NOTE(markn): This is a hack because 0-dim pytorch tensors are not iterable.
                # This occurs with batch size 1, because we still want to include the loss in that case.
                if output.dim() == 0:
                    output = output.unsqueeze(0)

                output = output.detach().cpu().numpy()
            
            for instance_output, batch_element in zip(instance_separated_output, output):
                instance_output[name] = batch_element
        
        return instance_separated_output


class ReactClassicalModel():

    def __init__(self, classifier, sample_wise_embedder:ReactSampleWiseFeatureEmbedder, label_embedder:ReactLabelEmbedder):
        self.classifier = classifier
        self._classifier_sanity_check() 
        self.sample_wise_embedder = sample_wise_embedder
        self.label_embedder = label_embedder

    def embed_texts_and_labels(self, texts:List[str], labels:List[str]) -> (torch.Tensor, torch.Tensor):
        """
        Takes in a list of texts and a list of labels, return embedded
            torch.Tensors.

        A easy way is to invoke ReactSampleWiseFeatureEmbedder
        """
        embedded_texts = self.sample_wise_embedder.embed_texts(texts)
        embedded_labels = self.label_embedder.embed_labels(labels)
        return embedded_texts, embedded_labels


    def _classifier_sanity_check(self) -> None:
        """
        Sanity check. The model that get plugged in to self.classifier
            should support .fit(X) and .predict(y)

        Will throw and error if that is not the case
        """
        try:
            assert(self.classifier.fit is not None)
            assert(self.classifier.predict is not None)
        except:
            print('AttributeError: your classifier for ReactClassicalModel should support .fit(X) and .predict(y)')
            raise AttributeError

    def extract_important_features(self, coefs, feature_names, k=10) -> dict:
        """
        Extract the most important features of the model.
        Inputs:
            lr_coef: coef_ from a lr model
            label_map: mapping from int label to str label (maybe I missremembered and its the inverse)
            feature_names: feature names corresponding to each dim
        Returns: Feature importances of shape=(no. classes, no. features).
        """
        out = {}
        # coefs = self.classifier.coef_
        label_map = self.label_embedder.get_inverse_label_mapping()

        # extract feature coefficients for each class
        for i, coef in enumerate(coefs):
            best_coef_indices = numpy.argsort(coef)[::-1]
            # display the top features for this class
            label_name = label_map[i] if coefs.shape[0] > 1 else label_map[1]
            #print('{} (Top {:,} features)'.format(label_name, k))
            out[label_name] = []
            for ndx in best_coef_indices[:k]:
                #print('{}: {:.3f}'.format(feature_names[ndx], coef[ndx]))
                out[label_name].append((feature_names[ndx], coef[ndx]))
            #print('')
        #print(out)
        return out

    def generate_multiclass_roc_curves(self, y_true, y_score, class_names=None):
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
        all_fpr = numpy.unique(numpy.concatenate([fpr for k, (fpr, tpr, _) in roc_curves.items()]))
        # Then interpolate all ROC curves at this points
        mean_tpr = numpy.zeros_like(all_fpr)
        for k, (fpr, tpr, _) in roc_curves.items():
            mean_tpr += numpy.interp(all_fpr, fpr, tpr)
        # finally average it
        mean_tpr /= n_classes
        roc_curves['Macro Average'] = (all_fpr, mean_tpr, None)
        return roc_curves

    def plot_roc_curves(self, curves, ax=None, zoom=False, width=18, legend_fontsize=7.5):
        """
        Plot ROC curve.
        """
        golden_ratio = 1.61803
        if ax is None:
            fig = plt.figure(figsize=(width, width / golden_ratio))
            ax = fig.add_axes([1, 1, 1, 1])
        ax.set_title('ROC curves')
        ax.set_ylabel("True Positive Rate")
        ax.set_ylim([-0.05, 1.05])
        ax.set_yticks(numpy.arange(0, 1, 0.1), minor=True)
        ax.set_xlabel("False Positive Rate")
        ax.set_xticks(numpy.arange(0, 1, 0.1), minor=True)
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

    