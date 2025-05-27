import torch
from torch.nn import functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import pipeline, set_seed
import tiktoken
from model import GPT2

N_EMBD = 768
N_BLOCK = 12
N_HEADS = 12
SEQ_LEN = 1024
DROPOUT = 0.1
HEAD_DIM = N_EMBD // N_HEADS 

# model = GPT2LMHeadModel.from_pretrained("gpt2") # 124M
model = GPT2(
    n_blocks=N_BLOCK,
    seq_len=SEQ_LEN,
    n_embd=N_EMBD,
    n_head=N_HEADS,
    vocab_size=50257,
    dropout=DROPOUT
)
model.load_state_dict(torch.load("myGPT2_from_pretrained.pth", weights_only=True))
model.eval()
model.to('cuda')
torch.manual_seed(42)
torch.cuda.manual_seed(42)
enc = tiktoken.get_encoding('gpt2')
# tokens = [15496, 11, 314, 1101, 257, 3303, 2746, 11] # "Hello, I'm a language model,"
tokens = enc.encode("Knight: This forest holds many secrets. Witch: Not all of them are meant for mortal ears. Knight:")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(5, 1) # (5, 8)
x = tokens.to('cuda')

# generate!
SEQ_LEN = 200
while x.size(1) < SEQ_LEN: # max_length=30
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text

for i in range(5):
    tokens = x[i, :SEQ_LEN].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)