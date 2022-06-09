"""
Defender Bert Model

Code written by Jonathan
"""
import torch
import numpy as np
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import RobertaTokenizer
from transformers import RobertaForSequenceClassification
from transformers import XLNetTokenizer
from transformers import XLNetForSequenceClassification
from transformers import GPT2Model
from transformers.models.bert.modeling_bert import BertEmbeddings
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings
from transformers import AutoTokenizer
from transformers import AutoModel



####################
# Model Definitions
####################


class BERTClassifier(torch.nn.Module):
    """
    Simple text classification model using a pretrained
    BERTSequenceClassifier model to tokenize, embed, and classify the input.
    """
    def __init__(self, pretrained_weights='bert-base-cased', max_seq_len=100, num_labels=2):
        super(BERTClassifier, self).__init__()

        self.max_seq_len = max_seq_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load BERT-base pretrained model
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        self.classifier = BertForSequenceClassification.from_pretrained(pretrained_weights,
                                                                        return_dict=True,
                                                                        num_labels=num_labels)

    def forward(self, text_list):
        """
        Define the forward pass.
        """
        inputs = self.tokenizer(text_list, padding=True, truncation=True,
                                max_length=self.max_seq_len, return_tensors='pt').to(self.device)
        return self.classifier(**inputs).logits

    def get_cls_repr(self, text_list):
        """
        Define the forward pass.
        """
        with torch.no_grad():
            inputs = self.tokenizer(text_list, padding=True, truncation=True,
                                    max_length=self.max_seq_len, return_tensors='pt').to(self.device)
        return self.classifier(**inputs, output_hidden_states=True).hidden_states[-1][:, 0, :] # the first hidden state in last layer

    def gradient(self, text, label, loss_fn=torch.nn.CrossEntropyLoss()):
        """
        Return gradients for this sample.
        """
        self.classifier.zero_grad()  # reset gradients
        pred = self.forward(text)  # forward pass
        label = label.to(self.device)
        loss = loss_fn(pred, label)  # compute loss
        loss.backward()  # backward pass
        gradients = [p.grad for p in self.classifier.parameters()]
        return gradients







