import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
import os

root_dir = os.path.dirname(os.path.abspath(__file__))

class FineWebEdu(Dataset):

    def __init__(self,split, n_chunks = None):

        
        if split == "val":
            
            dataset_loc = root_dir + "/data/val_tokenized.pt"         
        
        elif split == "train":
            pass
        
        self.data = torch.load(dataset_loc)

    @property
    def shape(self):
        return  self.data["tokens"].shape
    

    def __getitem__(self, index):
        
        return self.data["tokens"][index] , self.data["attention_masks"][index]

