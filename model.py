import torch
import torch.nn as nn
from torch.nn import functional as F

class AttentionHead(nn.Module):

    def __init__(self,head_dim,seq_len,d_embed):

        super().__init__()

        self.head_dim= head_dim
        self.seq_len = seq_len

        self.qkv = nn.Linear(d_embed,3*head_dim, bias= False)

        self.register_buffer("tril",torch.tril(torch.ones([seq_len,seq_len])))
    

    def forward(self,x):

        # calculate the qkv values
        qkv = self.qkv(x)

        #split
        q, k, v = qkv.split(self.head_dim, dim=-1)

        # calculate the multiplication
        wei = q@torch.transpose(k,-2,-1) / self.head_dim**0.5
        # mask out future values (-inf after softmax = 0)
        wei = wei.masked_fill(self.tril[:self.seq_len,:self.seq_len] == 0, float("-inf"))
        # softmax
        wei = F.softmax(wei,dim=-1)

        out = wei@v

        return out



class FeedForward(nn.Module):

    def __init__(self, d_embd, dropout):

        super().__init__()

        self.net = nn.Sequential
        (
            nn.Linear(d_embd,4*d_embd),
            nn.ReLU(),
            nn.Linear(4*d_embd,d_embd),
            nn.Dropout(dropout)
        )


class MultiHeadAttention(nn.Module):

    def __init__(self,num_heads,head_dim,seq_len,d_embed):
        super().__init__()

        self.heads = nn.ModuleList([AttentionHead(head_dim,seq_len,d_embed) for _ in range(num_heads)])
        self.proj = nn.Linear(d_embed,d_embed)
    
    def forward(self,x):

        out = torch.cat([ h(x) for h in self.heads],dim =-1 )
        out = self.proj(out)
        return out