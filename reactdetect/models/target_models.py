"""
Toxic comment classificaion (binary) models.
"""
import torch
import numpy as np


from transformers.models.bert.modeling_bert import BertEmbeddings
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import RobertaTokenizer
from transformers import RobertaForSequenceClassification
from transformers import XLNetTokenizer
from transformers import XLNetForSequenceClassification
from transformers import GPT2Model

stop_words = [
    "a", "about", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along",
    "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and",
    "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "back",
    "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can", "co",
    "con", "could", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg",
    "eight", "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every",
    "everyone", "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill", "find", "fire", "first",
    "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get",
    "give", "go", "had", "has", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon",
    "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "ie", "if", "in", "inc",
    "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least",
    "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most",
    "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "nevertheless", "next", "nine",
    "nobody", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others",
    "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps", "please", "put",
    "rather", "re", "same", "see", "serious", "several", "she", "should", "show", "side", "since", "sincere",
    "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still",
    "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence",
    "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too",
    "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very",
    "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter",
    "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who",
    "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your",
    "yours", "yourself", "yourselves", ">>>>"
]



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

class RoBERTaClassifier(torch.nn.Module):
    """
    Simple text classification model using
    a pretrained RoBERTaSequenceClassifier model to tokenize, embed, and classify
    the input.
    """
    def __init__(self, pretrained_weights='roberta-base', max_seq_len=100, num_labels=2):
        super(RoBERTaClassifier, self).__init__()

        self.max_seq_len = max_seq_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load RoBERTa-base pretrained model
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_weights)
        self.classifier = RobertaForSequenceClassification.from_pretrained(pretrained_weights,
                                                                           return_dict=True,
                                                                           num_labels=num_labels)

    def forward(self, text_list):
        """
        Define the forward pass.
        """
        inputs = self.tokenizer(text_list, padding=True, truncation=True,
                                max_length=self.max_seq_len, return_tensors='pt').to(self.device)
        return self.classifier(**inputs).logits

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

# class RoBERTaClassifier(torch.nn.Module):
#     """
#     Simple text classification model using
#     a pretrained RoBERTaSequenceClassifier model to tokenize, embed, and classify
#     the input.
#     """
#     def __init__(self, pretrained_weights='roberta-base', max_seq_len=100):
#         super(RoBERTaClassifier, self).__init__()

#         self.max_seq_len = max_seq_len
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         # load RoBERTa-base pretrained model
#         self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_weights)
#         self.classifier = RobertaForSequenceClassification.from_pretrained(pretrained_weights, return_dict=True)

#     def forward(self, text_list, labels=None, logits=False):
#         """
#         Define the forward pass.
#         """
#         inputs = self.tokenizer(text_list, padding=True, truncation=True,
#                                 max_length=self.max_seq_len, return_tensors='pt').to(self.device)
#         if labels is not None:
#             labels = labels.to(self.device)
        
#         outputs = self.classifier(**inputs, labels=labels)
#         result = outputs.logits if logits else outputs
#         return result

#     def gradient(self, text, label):
#         """
#         Return gradients for this sample.
#         """
#         self.classifier.zero_grad()  # clear gradients
#         output = self.forward(text, label)
#         output.loss.backward()
#         gradients = [p.grad for p in self.classifier.parameters()]
#         return gradients

#     def get_input_embeddings(self):
#         return self.classifier.roberta.embeddings.word_embeddings


class XLNetClassifier(torch.nn.Module):
    """
    Simple text classification model using a pretrained
    BERTSequenceClassifier model to tokenize, embed, and classify the input.
    """
    def __init__(self, pretrained_weights='xlnet-base-cased', max_seq_len=250, num_labels=2):
        super(XLNetClassifier, self).__init__()

        self.max_seq_len = max_seq_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load pretrained model
        self.tokenizer = XLNetTokenizer.from_pretrained(pretrained_weights)
        self.classifier = XLNetForSequenceClassification.from_pretrained(pretrained_weights,
                                                                         return_dict=True,
                                                                         num_labels=num_labels)

    def forward(self, text_list):
        """
        Define the forward pass.
        """
        inputs = self.tokenizer(text_list, padding=True, truncation=True,
                                max_length=self.max_seq_len, return_tensors='pt').to(self.device)
        return self.classifier(**inputs).logits

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


class UCLMRClassifier(torch.nn.Module):
    """
    UCL Machine Reading group Fake News Classifier.
    Input: 5K TF-IDF features for the headline, concatenated with
           5K TF-IDF features for the body, concatenated with
           cosine similarity between the two feature vectors.
    Model: 1 hidden layer with dimension 100 and ReLU activation.
    Output: 4 classes: agree, disagree, discuss, unrelated.
    Random batches
    Dropout: 0.4
    Learning rate: 0.001
    Optimizer: adam
    Loss_fn: crossentropy
    Weight decay: 1e-5
    Batch_size: 500
    Epochs: 1,000
    Max_norm: 5.0
    Trains on the entire training set, no validation set.
    """
    def __init__(self, tf_vectorizer, tfidf_vectorizer, n_hidden=100, n_classes=4, dropout=0.4):
        super(UCLMRClassifier, self).__init__()

        # save no. classes
        self.n_classes = n_classes

        # get device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # transforms the input, make sure max tokens is 5000
        self.tf_vectorizer = tf_vectorizer
        self.tfidf_vectorizer = tfidf_vectorizer
        assert self.tfidf_vectorizer.max_features == 5000
        assert self.tf_vectorizer.max_features == 5000

        # hidden layer
        self.fc1 = torch.nn.Linear(10001, n_hidden)
        self.fc2 = torch.nn.Linear(n_hidden, n_classes)

        # dropout layer
        self.d1 = torch.nn.Dropout(dropout)
        self.d2 = torch.nn.Dropout(dropout)

        # relu activation function
        self.relu = torch.nn.ReLU()

    def forward(self, text_input_list, logits=None):
        """
        `logits` only there for compatibility.
        """
        if type(text_input_list) != list:
            if type(text_input_list) == tuple:
                text_input_list = [text_input_list[0]]
            elif type(text_input_list) == str:
                text_input_list = [text_input_list]
            else:
                raise ValueError(f'invalid input {text_input_list}')

        # separate input into headline and body text
        token = '|||||'
        text_list = [x.split(token) for x in text_input_list]
        head_list, body_list = list(zip(*text_list))

        # transform
        head_tf = np.array(self.tf_vectorizer.transform(head_list).todense())
        body_tf = np.array(self.tf_vectorizer.transform(body_list).todense())

        head_tfidf = np.array(self.tfidf_vectorizer.transform(head_list).todense())
        body_tfidf = np.array(self.tfidf_vectorizer.transform(body_list).todense())

        # compute cosine similarity between head and body
        similarity = np.array([np.dot(head_tfidf[i], body_tfidf[i].T) for i in range(head_tfidf.shape[0])])

        # concatenate head and body features, shape=(batch_size, 10,001)
        feature_vec = np.hstack([head_tf, body_tf, similarity.reshape(-1, 1)])
        feature_vec = torch.tensor(feature_vec, dtype=torch.float32).to(self.device)

        # fully connected network
        x = self.fc1(feature_vec)
        x = self.relu(x)
        x = self.d1(x)

        x = self.fc2(x)
        x = self.d2(x)

        return x

    def gradient(self, text, label, loss_fn=torch.nn.CrossEntropyLoss()):
        """
        Return gradients for this sample.
        """
        self.zero_grad()  # clear gradients
        pred = self.forward(text)  # forward pass
        label = label.to(self.device)
        loss = loss_fn(pred, label)
        loss.backward()  # compute gradients
        gradients = [p.grad for p in self.parameters()]
        return gradients



################
# useful macros
################

def get_target_model(model_name, max_seq_len):
    """
    Return a new instance of the text classification model.
    """
    if model_name == 'roberta':
        model = RoBERTaClassifier(max_seq_len=max_seq_len)

    else:
        raise ValueError('Unknown model {}!'.format(model_name))

    return model

def load_target_model(target_model_name, dir_target_model, max_seq_len ,device):
    """
    Loads trained model weights and sets model to evaluation mode.
    """

    # load trained model
    model = get_target_model(model_name=target_model_name, max_seq_len=max_seq_len)
    model.load_state_dict(torch.load(os.path.join(dir_target_model, 'weights.pt'), map_location=device))
    model = model.to(device)
    model.eval()

    return model