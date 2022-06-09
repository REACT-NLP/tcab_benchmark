from typing import Dict, List, Set, Tuple
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.utils import shuffle as skl_shuffle
import numpy as np


class ReactDataLoader():

    def __init__(self, texts, labels, oversample=False, random_state=1):
        self._texts = texts 
        self._labels = labels
        self.random_state=random_state
        if oversample:
            self.oversample()
        
    def __getitem__(self, i:int) -> List[str]:
        """
        This is to make sure ReactDataLoader behaves like a tuple(texts, labels).
            note that __getitem__ do not return one pair of (text,label) 

        The only valid indexes are 0 and 1.
        """
        try:
            assert (i in [0,1])
        except:
            raise IndexError
        if i == 0:
            return [i for i in self._texts]
        else:
            return [i for i in self._labels]
        
    
    def oversample(self, wrapping=False):
        self._texts, self._labels = self._oversample(self._texts, self._labels, wrapping=wrapping)
        return self

    def _oversample(self,texts, labels, wrapping=False):
        if type(texts[0]) is str or wrapping==True:
            reshaped_texts = [[i] for i in texts] #needed by imblearn
            resampled_texts,resampled_labels =  RandomOverSampler(random_state=self.random_state).fit_resample([[i] for i in texts], labels)
            resampled_texts = [i[0] for i in resampled_texts] #de-wrap the list outside of texts
        else:
            reshaped_texts = texts
            resampled_texts,resampled_labels =  RandomOverSampler(random_state=self.random_state).fit_resample([i for i in texts], labels)
            resampled_texts = [i for i in resampled_texts] #de-wrap the list outside of texts
        
        return resampled_texts, resampled_labels

    
    def set_target_device(self, device) -> None:
        pass 

    def __repr__(self) -> str:
        return 'ReactDataLoaderObject: texts of len '+str(len(self._texts))+' || labels of len '+str(len(self._labels))+' ; class distribution: '+str(dict(Counter(self.get_labels())))

    def get_texts(self) -> List[str]:
        return self[0]

    def get_labels(self) -> List[str]:
        return self[1]




if __name__ == '__main__':
    rdl = ReactDataLoader([],[])