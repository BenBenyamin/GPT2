import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.amp import autocast

class GPT2Loss(nn.Module):

    def __init__(self):

        super().__init__()
    
    def forward(self,logits,targets):

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),targets.view(-1) , ignore_index=50256)

        return loss






