import torch
from torch import nn as nn
import torch.nn.functional as F

class GPT2Loss(nn.Module):

    def __init__(self):

        super().__init__()
    
    def forward(self,logits,tragets,attention_masks):

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),tragets.view(-1),reduction='none')
        mask = attention_masks.view(-1).float()

        loss = mask*loss
        loss = loss.sum()/mask.sum().clamp_min(1.0) ## Avoid dividing by zero

        return loss
