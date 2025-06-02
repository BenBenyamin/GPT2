import torch
import torch.nn as nn
from torch.nn import functional as F

class AttentionHead(nn.Module):
    """
    Implements a single attention head used in the transformer architecture.

    Args:
        head_dim (int): Dimension of each attention head.
        seq_len (int): Maximum sequence length.
        d_embed (int): Dimensionality of token embeddings.
        dropout (float): Dropout probability for attention scores.
    """
    def __init__(self, head_dim, seq_len, d_embed, dropout):
        super().__init__()
        self.head_dim = head_dim
        self.seq_len = seq_len

        self.q = nn.Linear(d_embed, head_dim, bias=True)
        self.k = nn.Linear(d_embed, head_dim, bias=True)
        self.v = nn.Linear(d_embed, head_dim, bias=True)

        self.register_buffer("tril", torch.tril(torch.ones([seq_len, seq_len])))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for a single attention head.

        Args:
            x (Tensor): Input tensor of shape (B, T, d_embed).

        Returns:
            Tensor: Output tensor after applying self-attention.
        """
        T = x.size(1)
        q = self.q(x).unsqueeze(1)
        k = self.k(x).unsqueeze(1)
        v = self.v(x).unsqueeze(1)

        attn_mask = self.tril[:T, :T].to(dtype=torch.bool)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout.p, is_causal=True
        )
        out = out.squeeze(1)
        return out


class MultiHeadAttention(nn.Module):
    """
    Combines multiple attention heads and applies a final linear projection.

    Args:
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension of each attention head.
        seq_len (int): Maximum sequence length.
        d_embed (int): Dimensionality of token embeddings.
        dropout (float): Dropout probability for projection.
    """
    def __init__(self, num_heads, head_dim, seq_len, d_embed, dropout):
        super().__init__()
        self.heads = nn.ModuleList([
            AttentionHead(head_dim, seq_len, d_embed, dropout) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(d_embed, d_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for multi-head attention.

        Args:
            x (Tensor): Input tensor of shape (B, T, d_embed).

        Returns:
            Tensor: Output tensor after concatenation and projection.
        """
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """
    Implements the feed-forward network used in transformer blocks.

    Args:
        d_embd (int): Dimensionality of embeddings.
        dropout (float): Dropout probability.
    """
    def __init__(self, d_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_embd, 4 * d_embd),
            nn.GELU(),
            nn.Linear(4 * d_embd, d_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Forward pass for feed-forward layer.

        Args:
            x (Tensor): Input tensor of shape (B, T, d_embd).

        Returns:
            Tensor: Output tensor.
        """
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Defines a single transformer block with attention, feed-forward, and layer norms.

    Args:
        n_head (int): Number of attention heads.
        n_embd (int): Embedding dimension.
        seq_len (int): Maximum sequence length.
        dropout (float): Dropout probability.
    """
    def __init__(self, n_head, n_embd, seq_len, dropout):
        super().__init__()
        head_dim = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_dim, seq_len, n_embd, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        """
        Forward pass for the transformer block.

        Args:
            x (Tensor): Input tensor of shape (B, T, n_embd).

        Returns:
            Tensor: Output tensor after attention and feed-forward network.
        """
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT2(nn.Module):
    """
    GPT-2 Language Model.

    Args:
        n_blocks (int): Number of transformer blocks.
        seq_len (int): Maximum sequence length.
        n_embd (int): Embedding dimension.
        n_head (int): Number of attention heads.
        vocab_size (int): Size of the vocabulary.
        dropout (float): Dropout probability.
    """
    def __init__(self, n_blocks, seq_len, n_embd, n_head, vocab_size, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.n_embd = n_embd

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(seq_len, n_embd)

        blocks = [
            TransformerBlock(n_head, n_embd, seq_len, dropout) for _ in range(n_blocks)
        ]
        self.blocks = nn.Sequential(*blocks)

        self.final_ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding_table.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Custom weight initialization for linear and embedding layers.
        """
        std = 1 / self.n_embd ** 0.5
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, tokens):
        """
        Forward pass for the GPT-2 model.

        Args:
            tokens (Tensor): Input tensor of token indices with shape (B, T).

        Returns:
            Tensor: Logits over the vocabulary for each token position.
        """
        B, T = tokens.shape
        token_embeddings = self.token_embedding_table(tokens)  # (B, T, n_embd)
        pos_embd = self.position_embedding_table(torch.arange(T, device=tokens.device))
        x = token_embeddings + pos_embd
        x = self.blocks(x)
        x = self.final_ln(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits
