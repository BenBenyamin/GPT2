import torch
from torch.nn import functional as F

import time

from model import GPT2

from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from dataset import FineWebEdu
from optimizer import CosineSchedulerWithWarmup
from loss import GPT2Loss

N_EMBD = 768
N_BLOCK = 12
N_HEADS = 12
SEQ_LEN = 1024
DROPOUT = 0.1
VOCAB_SIZE = 50257
HEAD_DIM = N_EMBD // N_HEADS 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE  = 8
EPOCHS = 1

model = GPT2(
    n_blocks=N_BLOCK,
    seq_len=SEQ_LEN,
    n_embd=N_EMBD,
    n_head=N_HEADS,
    vocab_size=VOCAB_SIZE,
    dropout=DROPOUT
).to(DEVICE)

# Load dataset
val_dataset = FineWebEdu("val")

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle= True,
)


optimizer = torch.optim.AdamW(params=model.parameters(),lr=1e-3)
loss_crit = GPT2Loss()

scaler = GradScaler('cuda')

for epoch in range(EPOCHS):

    model.train()

    for batch_idx, (tokens, targets) in enumerate(val_loader):
        
        start_time = time.time() *1000

        tokens, targets = tokens.to(DEVICE),  targets.to(DEVICE)

        with autocast(device_type='cuda'):
            logits = model(tokens)
            loss = loss_crit(logits, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        end_time = time.time() *1000
        elapsed = end_time - start_time

        total_tokens = tokens.numel()  # total number of elements in the tensor
        tokens_per_sec = total_tokens / (elapsed / 1000)

        print(f"EPOCH {epoch} Batch {batch_idx}/{len(val_loader)} - Loss: {loss.item():.4f} - "
              f"Step Time: {elapsed:.2f} ms - Tokens/sec: {tokens_per_sec:.2f}")
