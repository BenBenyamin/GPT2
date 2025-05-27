import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
import os

root_dir = os.path.dirname(os.path.abspath(__file__))

class FineWebEdu(Dataset):

    def __init__(self,split, agent_num ,n_chunks = None):

        if split == "val" or n_chunks is None:
            
            self.data = torch.load("val_tokenized.pt")     
        
        elif split == "train":
            input_ids = []
            masks = []
            self.data = {}

            for i in range (agent_num*n_chunks,(agent_num+1)*n_chunks):
                data = torch.load(f"train_tokenized_{i}.pt")
                input_ids.append(data["input_ids"])
                masks.append(data["attention_mask"])


            self.n_chunks = n_chunks

            self.data["input_ids"] = torch.cat(input_ids, dim=0)
            self.data["attention_mask"] = torch.cat(masks,dim=0)
                

    @property
    def shape(self):
        return  self.data["input_ids"].shape
    

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):

        tokens = self.data["input_ids"][index]
        
        # Shift tokens for targets within the same sequence
        inputs = tokens[:-1]
        targets = tokens[1:]

        return inputs, targets
