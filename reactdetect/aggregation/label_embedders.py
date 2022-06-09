from __future__ import annotations
import torch


class ReactLabelEmbedder():

    def __init__(self, mapping, class_names=None, cuda_device=None):
        self.mapping = mapping    
        self.cuda_device = cuda_device
        self.class_names = class_names
    
    def embed_labels(self, labels: list) -> torch.Tensor:
        
        out = []
        for l in labels:
            try:
                out.append(self.mapping[l])
            except:
                print('offending label: ',l)
                print('labels: ', labels)
        if self.cuda_device is not None:
            return torch.tensor(out).to(self.cuda_device)
        return torch.tensor(out)

    def get_num_labels(self) -> int:

        return len(set(self.mapping.keys()))

    def get_label_mapping(self):
        return self.mapping

    def get_class_names(self):
        return self.class_names
    
    def get_inverse_label_mapping(self):
        inverse_mapping = {}
        for key, value in self.mapping.items():
            inverse_mapping[ int(value) ] = key 
        return inverse_mapping

    def set_device(self, cuda_device) -> ReactLabelFeatureEmbedder:
        self.cuda_device = cuda_device 
        return self


if __name__ == '__main__':
    rle = ReactLabelEmbedder({'happy':1,'sad':0})