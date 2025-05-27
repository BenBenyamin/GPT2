import torch
from torch.nn import functional as F

import time

from model import GPT2

from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from dataset import FineWebEdu
from optimizer import CosineSchedulerWithWarmup
from loss import GPT2Loss

torch.set_float32_matmul_precision('high')

LR = 6e-4

N_EMBD = 768
N_BLOCK = 12
N_HEADS = 12
SEQ_LEN = 1024
DROPOUT = 0.1
VOCAB_SIZE = 50304 # orignal : 50257 ,  more divisible by 2
HEAD_DIM = N_EMBD // N_HEADS 

TOKEN_PER_BATCH = 524288
BATCH_SIZE  = 32
BATCH_ITER = TOKEN_PER_BATCH // (1024 *BATCH_SIZE)
EPOCHS = 1

# Load dataset
train_dataset = FineWebEdu("train", agent_num=1, n_chunks=24//2)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle= True,
)

LR_WARMUP_STEPS = 375e6 // TOKEN_PER_BATCH
TOTAL_STEPS = len(train_loader) // BATCH_ITER * EPOCHS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = GPT2(
    n_blocks=N_BLOCK,
    seq_len=SEQ_LEN,
    n_embd=N_EMBD,
    n_head=N_HEADS,
    vocab_size=VOCAB_SIZE,
    dropout=DROPOUT
).to(DEVICE)

model = torch.compile(model)


# optimizer = torch.optim.AdamW(params=model.parameters(),lr=1e-3)
optimizer = CosineSchedulerWithWarmup(
    named_params=model.named_parameters(),
    lr = LR, 
    warmup_steps=LR_WARMUP_STEPS,
    total_steps=TOTAL_STEPS
    )
loss_crit = GPT2Loss()

scaler = GradScaler('cuda')

for epoch in range(EPOCHS):

    model.train()
    optimizer.zero_grad()
    acc_loss = 0.0
    start_time = time.time() *1000

    for batch_idx, (tokens, targets) in enumerate(train_loader):
        
        tokens, targets = tokens.to(DEVICE),  targets.to(DEVICE)

        with autocast(device_type='cuda'):
            logits = model(tokens)
            loss = loss_crit(logits, targets) / BATCH_ITER

        acc_loss += loss.item() # remove from comp graph
        scaler.scale(loss).backward()

        #Grad accumulation
        if ((batch_idx+1)%BATCH_ITER == 0):
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            elapsed = (time.time()*1000) - start_time 
            total_tokens = tokens.numel() *BATCH_ITER
            tokens_per_sec = total_tokens / (elapsed/1000)
            print(
                f"EPOCH {epoch} Batch {(batch_idx+1)//BATCH_ITER}/{len(train_loader)//BATCH_ITER}  "
                f"Loss: {acc_loss:.4f}  Norm: {norm:.2f}  LR: {optimizer.lr:.4e}  "
                f"Step Time: {elapsed:.1f} ms  {tokens_per_sec:.0f} tok/s"
            )
            start_time = time.time() *1000
            acc_loss = 0.0
