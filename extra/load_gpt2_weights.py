from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from model import *

"""
This script converts pretrained GPT-2 weights from Hugging Face into this GPT-2 implementation.
It initializes the custom model with the same architecture as GPT2-small and copies over:
  - token and position embeddings
  - attention head weights (Q, K, V projections)
  - attention output projections
  - MLP layers (feed-forward network)
  - LayerNorm parameters

This was done for debugging purposes.
"""

# Model configuration matching GPT2-small
N_EMBD = 768
N_BLOCK = 12
N_HEADS = 12
SEQ_LEN = 1024
DROPOUT = 0.1
HEAD_DIM = N_EMBD // N_HEADS

# Load pretrained GPT-2 model and tokenizer from Hugging Face
hf_model = GPT2LMHeadModel.from_pretrained('gpt2')
hf_model.eval()
hf_state_dict = hf_model.state_dict()

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Initialize custom GPT-2 model
my_model = GPT2(
    n_blocks=N_BLOCK,
    seq_len=SEQ_LEN,
    n_embd=N_EMBD,
    n_head=N_HEADS,
    vocab_size=len(tokenizer),
    dropout=DROPOUT
)
my_model.eval()

# Copy token embedding weights
my_model.token_embedding_table.weight.data.copy_(
    hf_state_dict['transformer.wte.weight']
)

# Copy position embedding weights
my_model.position_embedding_table.weight.data.copy_(
    hf_state_dict['transformer.wpe.weight']
)

# Weight tying between embeddings and lm_head
my_model.lm_head.weight = my_model.token_embedding_table.weight

# Copy weights for each transformer block
for i in range(N_BLOCK):
    qkv_w = hf_state_dict[f'transformer.h.{i}.attn.c_attn.weight']
    qkv_b = hf_state_dict[f'transformer.h.{i}.attn.c_attn.bias']

    # Split QKV weights and biases
    q_w, k_w, v_w = qkv_w.chunk(3, dim=1)
    q_b, k_b, v_b = qkv_b.chunk(3, dim=0)

    for h in range(N_HEADS):
        start = h * HEAD_DIM
        end = (h + 1) * HEAD_DIM

        # Transpose slices to match Linear layer shape
        qw = q_w[:, start:end].T 
        kw = k_w[:, start:end].T
        vw = v_w[:, start:end].T

        qb = q_b[start:end]
        kb = k_b[start:end]
        vb = v_b[start:end]

        my_model.blocks[i].sa.heads[h].q.weight.data.copy_(qw)
        my_model.blocks[i].sa.heads[h].q.bias.data.copy_(qb)
        my_model.blocks[i].sa.heads[h].k.weight.data.copy_(kw)
        my_model.blocks[i].sa.heads[h].k.bias.data.copy_(kb)
        my_model.blocks[i].sa.heads[h].v.weight.data.copy_(vw)
        my_model.blocks[i].sa.heads[h].v.bias.data.copy_(vb)

    # Copy attention output projection
    my_model.blocks[i].sa.proj.weight.data.copy_(
        hf_state_dict[f'transformer.h.{i}.attn.c_proj.weight'].T
    )
    my_model.blocks[i].sa.proj.bias.data.copy_(
        hf_state_dict[f'transformer.h.{i}.attn.c_proj.bias']
    )

    # Copy MLP (feedforward network) weights
    my_model.blocks[i].ffwd.net[0].weight.data.copy_(
        hf_state_dict[f'transformer.h.{i}.mlp.c_fc.weight'].T
    )
    my_model.blocks[i].ffwd.net[0].bias.data.copy_(
        hf_state_dict[f'transformer.h.{i}.mlp.c_fc.bias']
    )
    my_model.blocks[i].ffwd.net[2].weight.data.copy_(
        hf_state_dict[f'transformer.h.{i}.mlp.c_proj.weight'].T
    )
    my_model.blocks[i].ffwd.net[2].bias.data.copy_(
        hf_state_dict[f'transformer.h.{i}.mlp.c_proj.bias']
    )

    # Copy LayerNorm weights
    my_model.blocks[i].ln1.weight.data.copy_(
        hf_state_dict[f'transformer.h.{i}.ln_1.weight']
    )
    my_model.blocks[i].ln1.bias.data.copy_(
        hf_state_dict[f'transformer.h.{i}.ln_1.bias']
    )
    my_model.blocks[i].ln2.weight.data.copy_(
        hf_state_dict[f'transformer.h.{i}.ln_2.weight']
    )
    my_model.blocks[i].ln2.bias.data.copy_(
        hf_state_dict[f'transformer.h.{i}.ln_2.bias']
    )

# Copy final LayerNorm
my_model.final_ln.weight.data.copy_(
    hf_state_dict['transformer.ln_f.weight']
)
my_model.final_ln.bias.data.copy_(
    hf_state_dict['transformer.ln_f.bias']
)

# Optionally save the converted weights
# torch.save(my_model.state_dict(), "myGPT2_from_pretrained.pth")
