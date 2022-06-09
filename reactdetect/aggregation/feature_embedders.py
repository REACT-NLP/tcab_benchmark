from __future__ import annotations
import torch
from tqdm import tqdm 
import joblib
import os
from allennlp.nn.util import move_to_device

def wrap_tqdm(iterable, verbose:bool)->tqdm:
    """
    make iterable tqdm if verbose
    """
    if verbose:
        return tqdm(iterable)
    else:
        return iterable


class ReactBaseFeatureEmbedder():
    """
    Base embedder that handles io, caching, etc.
        agnostic of samplewise/tokenwise, etc
    """
    def __init__(self, use_cache=True,cuda_device=None, verbose=False) -> None:
        self.cache = {}
        self.use_cache = use_cache 
        self.cuda_device=cuda_device
        self.verbose=verbose
        self._set_cuda_datatypes(self.cuda_device)


    def _know(self, sent) -> bool:
        """
        check if a sentence is already embedded and stored in self.cache
        """
        return sent in self.cache.keys()

    def _remember(self, sent, feature_repr) -> None:
        """
        remember a sentence's representation in self.cache.
        """
        if not self.use_cache:
            return
        self.cache[sent] = move_to_device(feature_repr, torch.device('cpu'))

    def _fetch(self, sent):
        """
        returns cached representation of a cached sentence
        """
        return move_to_device(self.cache[sent], self.cuda_device)

    def get_output_dim(self)->int:
        """
        return output dimention of feature size for a single sample
        """
        raise NotImplementedError

    def _set_cuda_datatypes(self, cuda_device):
        """
        using alias to determine tensor types
            modify the block to add stuff of your choice
        """
        if str(self.cuda_device) in ['cpu', 'None']:
            self.FloatTensor = torch.FloatTensor
            self.BoolTensor = torch.BoolTensor
        else:
            self.FloatTensor = torch.cuda.FloatTensor
            self.BoolTensor = torch.cuda.BoolTensor

    def set_device(self, cuda_device) -> ReactBaseFeatureEmbedder:
        self.cuda_device = cuda_device 
        self._set_cuda_datatypes(self.cuda_device)
        return self
    

    def set_verbose(self, verbose:bool) -> ReactBaseFeatureEmbedder:
        self.verbose = verbose
        return self

    def get_cache_file_dir(self)->str:
        """
        return a file dir that will be used for caching
        """
        raise NotImplementedError
    
    def save_cache(self) -> bool:
        """
        save current cache
        """
        try:
            joblib.dump(self.cache, self.get_cache_file_dir())
            return True 
        except FileNotFoundError:
            self.vprint('cache directory or file DNE, creating a new one for you')
            if not os.path.exists(os.path.dirname(self.get_cache_file_dir())):
                os.mkdir(os.path.dirname(self.get_cache_file_dir()))
            joblib.dump(self.cache, self.get_cache_file_dir())
            return True
        except Exception as e:
            print('ReactrEmbedder: ',e, ' when saving caches')
            return False

    def load_cache(self):
        """
        load from some cache dir
        """
        try:
            self.cache = joblib.load(self.get_cache_file_dir())
            self.vprint('cache loaded from '+self.get_cache_file_dir())
        except FileNotFoundError:
            self.vprint('cache directory or file DNE, creating a new one for you')
            if not os.path.exists(os.path.dirname(self.get_cache_file_dir())):
                os.mkdir(os.path.dirname(self.get_cache_file_dir()))
            self.cache = {}
            joblib.dump(self.cache, self.get_cache_file_dir())
        except NotImplementedError:
            self.vprint('no get_cache_file_dir() implemented, cache will be a dynamic dict that is newly constructed')
        
        return self

    def save_cache_to(self, cache_file_dir) -> bool:
        """
        save current cache
        """
        try:
            joblib.dump(self.cache, cache_file_dir)
            return True 
        except FileNotFoundError:
            self.vprint('cache directory or file DNE, creating a new one for you')
            if not os.path.exists(os.path.dirname(cache_file_dir)):
                os.mkdir(os.path.dirname(cache_file_dir))
            joblib.dump(self.cache, cache_file_dir)
            return True
        except Exception as e:
            print('ReactrEmbedder: ',e, ' when saving caches')
            return False

    def load_cache_from(self, cache_file_dir):
        """
        load from some cache dir
        """
        try:
            self.cache = joblib.load(cache_file_dir)
            self.vprint('cache loaded from '+cache_file_dir)
        except FileNotFoundError:
            self.vprint('cache directory or file DNE, creating a new one for you')
            if not os.path.exists(os.path.dirname(cache_file_dir)):
                os.mkdir(os.path.dirname(cache_file_dir))
            self.cache = {}
            joblib.dump(self.cache, cache_file_dir)
        except NotImplementedError:
            self.vprint('no get_cache_file_dir() implemented, cache will be a dynamic dict that is newly constructed')
        
        return self

    def vprint(self, *args, **kw):
        """
        print function that can be tuned off with self.verbose flag.
            should've just use logger, probably..
        """
        if self.verbose:
            print(*args, **kw)

    def get_cache_file_dir(self, cache_dir:str=None):
        """
        implement this to return a reasonable working dir to save your caches for further usage
        """
        return os.path.join(os.getcwd(), 'react-embedder-caches',type(self).__name__+'.cache.joblib')

    def get_feature_names(self):
        """
        Implement this to return names of each feature dimension to enable analysis
        """
        raise NotImplementedError

    def get_embedder_name(self):
        """
        Return the name of leaf level Child class of this instance
        useful when e.g. determining what directory to save cache, logging, etc
        """
        return type(self).__name__

class ReactTokenWiseFeatureEmbedder(ReactBaseFeatureEmbedder):

    def __init__(self, use_cache=True,cuda_device=None, verbose=False) -> None:
        super().__init__(use_cache=use_cache, cuda_device=cuda_device, verbose=verbose)
    
    def embed_texts(self, samples:list, force_recompute= False):
        """
        return embedded text in the form of [batch_size, self.get_output_dim()]
        """
        with torch.set_grad_enabled(True):
            output = []
            saved_encodings = 0
            err_count = 0
            
            self.vprint('feature_embedder tokenwise is embedding ',len(samples),' samples')
            for sample in wrap_tqdm(samples,self.verbose):
                if not self._know(sample) or force_recompute:
                    try:
                        encoded = (self.FloatTensor(self.embed_text(sample)[0]),self.BoolTensor(self.embed_text(sample)[1]))
                        self._remember(sample, encoded)
                    except Exception as e:
                        self.vprint('an err occured with sample: ', samples,' with err: ',e)
                        encoded = (torch.zeros(self.get_output_dim()).to(self.cuda_device),self.BoolTensor(self.embed_text('hello world')[1]))
                        err_count += 1 #(zhouhanx): should let the err be noticed even when using cache, do not save this
                    
                else:
                    encoded = self._fetch(sample)
                    saved_encodings += 1
                output.append(encoded)
            self.vprint('feature_embedder tokenwise done, you saved ', saved_encodings,' operations. err count: ', err_count)
            
            
            embedded_texts = torch.stack([i[0] for i in output]) #because each "encoded" is a tuple(repr, mask)
            masks = torch.stack([i[1] for i in output])

            if self.cuda_device is not None:
                embedded_texts, masks = embedded_texts.to(self.cuda_device), masks.to(self.cuda_device)
        
        return embedded_texts,masks

    


class ReactSampleWiseFeatureEmbedder(ReactBaseFeatureEmbedder):

    def __init__(self, use_cache=True, cuda_device=None, verbose=False)-> None:
        super().__init__(use_cache=use_cache, cuda_device=cuda_device, verbose=verbose)

    def embed_texts(self, samples:list, force_recompute=False):
        """
        return embedded text in the form of [batch_size, self.get_output_dim()]
        """
        with torch.set_grad_enabled(True):
            output = []
            saved_encodings = 0
            err_count = 0
            self.vprint('feature_embedder samplewise is embedding ',len(samples),' samples')
            for sample in wrap_tqdm(samples, verbose=self.verbose):
                if not self._know(sample) or force_recompute:
                    try:
                        encoded = self.FloatTensor(self.embed_text(sample))
                        self._remember(sample, encoded)
                    except Exception as e:
                        # print(e)
                        # import sys
                        # sys.exit()
                        self.vprint('an err occured with sample: ', samples,' with err: ',e)
                        encoded = torch.zeros(self.get_output_dim()).to(self.cuda_device)
                        err_count += 1 #(zhouhanx): should let the err be noticed even when using cache, do not save this
                    
                else:
                    encoded = self._fetch(sample)
                    saved_encodings += 1
                output.append(encoded)
            self.vprint('feature_embedder samplewise done, you saved ', saved_encodings,' operations. err count: ', err_count)
            
            output = torch.stack(output)
            if self.cuda_device is not None:
                output = output.to(self.cuda_device)
            else:
                pass
        return output

    def get_feature_names(self):
        raise NotImplementedError

if __name__ == '__main__':
    rtfe = ReactTokenWiseFeatureEmbedder()
    rsfe = ReactSampleWiseFeatureEmbedder()


    
    
