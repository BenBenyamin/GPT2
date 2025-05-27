import torch
import torch.nn as nn
from torch.nn import functional as F

class AttentionHead(nn.Module):

    def __init__(self,head_dim,seq_len,d_embed,dropout):

        super().__init__()

        self.head_dim= head_dim
        self.seq_len = seq_len

        self.q = nn.Linear(d_embed,head_dim, bias= True)
        self.k = nn.Linear(d_embed,head_dim, bias= True)
        self.v = nn.Linear(d_embed,head_dim, bias= True)
        

        self.register_buffer("tril",torch.tril(torch.ones([seq_len,seq_len])))

        self.dropout = nn.Dropout(dropout)
    

    def forward(self,x):

        T = x.size(1)

        # calculate the qkv values
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # calculate the multiplication
        wei = q@torch.transpose(k,-2,-1) / self.head_dim**0.5
        # mask out future values (-inf after softmax = 0)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float("-inf"))
        # softmax
        wei = F.softmax(wei,dim=-1)
        wei = self.dropout(wei)

        out = wei@v

        return out



class MultiHeadAttention(nn.Module):

    def __init__(self,num_heads,head_dim,seq_len,d_embed,dropout):
        super().__init__()

        self.heads = nn.ModuleList([AttentionHead(head_dim,seq_len,d_embed,dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(d_embed,d_embed)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):

        out = torch.cat([h(x) for h in self.heads],dim =-1 )
        out = self.proj(out)
        out = self.dropout(out)
        return out
    

class FeedForward(nn.Module):

    def __init__(self, d_embd, dropout):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_embd,4*d_embd),
            nn.GELU(),
            nn.Linear(4*d_embd,d_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self,x):

        return self.net(x)

class TransformerBlock(nn.Module):
        
    def __init__(self, n_head , n_embd , seq_len, dropout):

        super().__init__()
        head_dim = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_dim,seq_len, n_embd,dropout)
        self.ffwd = FeedForward(n_embd,dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self,x):

        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x)) 

        return x

class GPT2(nn.Module):

    def __init__(self, n_blocks ,seq_len,n_embd, n_head,vocab_size,dropout):
        
        super().__init__()

        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.n_embd = n_embd

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(seq_len, n_embd)

        blocks = [TransformerBlock(n_head , n_embd , seq_len,dropout) for _ in range(n_blocks)]

        self.blocks = nn.Sequential(
            *blocks,
        )

        self.final_ln = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(n_embd , vocab_size, bias=False)

        self.lm_head.weight = self.token_embedding_table.weight

        self.apply(self._init_weights)

    def _init_weights(self,module):
        
        std = 1/self.n_embd**0.5
        if (isinstance(module,nn.Linear)):
            torch.nn.init.normal_(module.weight,mean = 0.0, std = std)
            if (module.bias is not None):
                torch.nn.init.zeros_(module.bias)
        
        elif (isinstance(module,nn.Embedding)):
            torch.nn.init.normal_(module.weight,mean = 0.0, std = std)

    def forward(self,tokens):
        
        B,T = tokens.shape

        token_embeddings = self.token_embedding_table(tokens) # (B,T,C)
        pos_embd = self.position_embedding_table(torch.arange(T,device=tokens.device))
        x = token_embeddings + pos_embd
        x = self.blocks(x)
        x = self.final_ln(x)
        logits  = self.lm_head(x) # (B,T,N_EMBD)

        # logits = logits.view(B*T,self.vocab_size)

        return logits

