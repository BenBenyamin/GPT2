from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from model import *

# Model configuration matching GPT2-small
N_EMBD = 768
N_BLOCK = 12
N_HEADS = 12
SEQ_LEN = 1024
DROPOUT = 0.1
HEAD_DIM = N_EMBD // N_HEADS 

# Load pretrained GPT2 weights from Hugging Face
hf_model = GPT2LMHeadModel.from_pretrained('gpt2')
hf_model.eval()
hf_state_dict = hf_model.state_dict()

# Initialize custom model
my_model = GPT2(
    n_blocks=N_BLOCK,
    seq_len=SEQ_LEN,
    n_embd=N_EMBD,
    n_head=N_HEADS,
    vocab_size=50257,
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

# Copy output projection weights
# my_model.lm_head.weight.data.copy_(
#     hf_state_dict['lm_head.weight']
# )

my_model.lm_head.weight = my_model.token_embedding_table.weight

# Loop over all transformer blocks
for i in range(N_BLOCK):
    qkv_w = hf_state_dict[f'transformer.h.{i}.attn.c_attn.weight']  # (768, 2304)
    qkv_b = hf_state_dict[f'transformer.h.{i}.attn.c_attn.bias']    # (2304,)

    # Split into q, k, v along in_features
    q_w, k_w, v_w = qkv_w.chunk(3, dim=1)  # Each is (768, 768)
    q_b, k_b, v_b = qkv_b.chunk(3, dim=0)  # Each is (768,)

    for h in range(N_HEADS):
        start = h * HEAD_DIM
        end = (h + 1) * HEAD_DIM

        # Slice along in_features (dim=1), then transpose for PyTorch Linear
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

    my_model.blocks[i].sa.proj.weight.data.copy_(
        hf_state_dict[f'transformer.h.{i}.attn.c_proj.weight'].T
    )
    my_model.blocks[i].sa.proj.bias.data.copy_(
        hf_state_dict[f'transformer.h.{i}.attn.c_proj.bias']
    )
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

# Load final LayerNorm
my_model.final_ln.weight.data.copy_(
    hf_state_dict['transformer.ln_f.weight']
)
my_model.final_ln.bias.data.copy_(
    hf_state_dict['transformer.ln_f.bias']
)

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Input prompt
prompt = "Hello GPT2!\n"
print(f"The prompt is {prompt}")
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

# Greedy generation loop
def generate(model, input_ids, max_new_tokens=500):
    model.eval()
    for _ in range(max_new_tokens):
        input_ids_trimmed = input_ids[:, -model.seq_len:]
        with torch.no_grad():
            logits = model(input_ids_trimmed)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)
    return input_ids

my_model.eval()
# Generate text using Hugging Face model
with torch.no_grad():
    hf_output = hf_model.generate(input_ids, max_new_tokens=500, do_sample=False)
hf_text = tokenizer.decode(hf_output[0], skip_special_tokens=True)
print("\nHugging Face output:")
print(hf_text)

# Generate text using custom model
generated_ids = generate(my_model, input_ids.clone(), max_new_tokens=500)
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("\nCustom model output:")
print(generated_text)




