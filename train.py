import torch
from torch.nn import functional as F

import time

from model import GPT2

from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from dataset import FineWebEdu
from optimizer import CosineSchedulerWithWarmup
from loss import GPT2Loss

from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import os

writer = SummaryWriter(log_dir="runs/gpt2")
torch.set_float32_matmul_precision('high')
# Set seed so each agent has the same model at first
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

AGENT_NUM = int(os.environ.get('LOCAL_RANK', 0))
TOTAL_AGENTS = int(os.environ.get('WORLD_SIZE', 1))
IS_DDP = TOTAL_AGENTS > 1
MASTER_PROCESS = AGENT_NUM == 0


if IS_DDP:
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(AGENT_NUM)

def log(str):
    if (MASTER_PROCESS): print(str)

log(f"Number of agents = {TOTAL_AGENTS}")


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
BATCH_ITER = TOKEN_PER_BATCH // (1024 *BATCH_SIZE *TOTAL_AGENTS)
EPOCHS = 1

# Load dataset
train_dataset = FineWebEdu("train", agent_num=AGENT_NUM, n_chunks=24//TOTAL_AGENTS)
val_dataset = FineWebEdu("val")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle= True,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    pin_memory=True
)



LR_WARMUP_STEPS = 375e6 // TOKEN_PER_BATCH
TOTAL_STEPS = len(train_loader) // BATCH_ITER * EPOCHS

DEVICE = f'cuda:{AGENT_NUM}' if IS_DDP else 'cuda' if torch.cuda.is_available() else 'cpu'

model = GPT2(
    n_blocks=N_BLOCK,
    seq_len=SEQ_LEN,
    n_embd=N_EMBD,
    n_head=N_HEADS,
    vocab_size=VOCAB_SIZE,
    dropout=DROPOUT
).to(DEVICE)

if (IS_DDP):
    
    model = DDP(model, device_ids=[AGENT_NUM])

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

model.train()
optimizer.zero_grad()

for epoch in range(EPOCHS):

    acc_loss = 0.0
    start_time = time.time() *1000

    for batch_idx, (tokens, targets) in enumerate(train_loader):
        
        tokens, targets = tokens.to(DEVICE),  targets.to(DEVICE)

        with autocast(device_type='cuda'):
            logits = model(tokens)
            loss = loss_crit(logits, targets) / BATCH_ITER 

        acc_loss += loss.item() # remove from comp graph
        if (IS_DDP):
            model.require_backward_grad_sync = ((batch_idx + 1) % BATCH_ITER == 0)

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
            log(
                f"EPOCH {epoch} Batch {(batch_idx+1)//BATCH_ITER}/{len(train_loader)//BATCH_ITER}  "
                f"Loss: {acc_loss:.4f}  Norm: {norm:.2f}  LR: {optimizer.lr:.4e}  "
                f"Step Time: {elapsed:.1f} ms  {tokens_per_sec:.0f} tok/s"
            )
            if (MASTER_PROCESS):
                writer.add_scalar("Train Loss",acc_loss,batch_idx*(epoch+1))
                writer.add_scalar("Tokens/Sec",tokens_per_sec,batch_idx*(epoch+1))
                writer.add_scalar("Learning Rate", optimizer.lr,batch_idx*(epoch+1))
            start_time = time.time() *1000
            acc_loss = 0.0

if IS_DDP:
    dist.destroy_process_group()