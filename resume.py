import torch
from torch.nn import functional as F

import time

from model import GPT2

from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from dataset import FineWebEdu
from optimizer import CosineSchedulerWithWarmup
from loss import GPT2Loss
from utils import *

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

# -----------------------
# Model Hyperparameters
# -----------------------
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
EPOCHS = 4
# -----------------------

# Logging / generating text
# -----------------------
VAL_LOG_BATCHES = 500 # How often to log
VAL_N_BATCHES = 500 # How many batches to evaluate on
LOG_FILE = "generation_log.txt"
# ------------------------

# # Load dataset
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

# Optimizer Settings
# -----------------------
LR_WARMUP_STEPS = 375e6 // TOKEN_PER_BATCH
TOTAL_STEPS = len(train_loader) // BATCH_ITER * EPOCHS
# -----------------------

DEVICE = f'cuda:{AGENT_NUM}' if IS_DDP else 'cuda' if torch.cuda.is_available() else 'cpu'

model = GPT2(
    n_blocks=N_BLOCK,
    seq_len=SEQ_LEN,
    n_embd=N_EMBD,
    n_head=N_HEADS,
    vocab_size=VOCAB_SIZE,
    dropout=DROPOUT
).to(DEVICE)

## Change here to load the checkpoint
checkpoint = torch.load("checkpoints/2_checkpoint.pt")
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Agent {AGENT_NUM} loaded the model")


if (IS_DDP):
    
    model = DDP(model, device_ids=[AGENT_NUM])

model = torch.compile(model)

optimizer = CosineSchedulerWithWarmup(
    named_params=model.named_parameters(),
    lr = LR, 
    warmup_steps=LR_WARMUP_STEPS,
    total_steps=TOTAL_STEPS
    )

# This is for the cosine scheduler , ignore if not applicable
# -----------------------
optimizer.step_num = checkpoint["step"] // BATCH_ITER +1
optimizer.set_next_lr()
log(f"Optimizer lr : {optimizer.lr}")
# -----------------------


loss_crit = GPT2Loss()
val_loss_tracker = ValLossTracker(
    val_loader=val_loader,
    num_of_batches= VAL_N_BATCHES,
)

scaler = GradScaler('cuda')

model.train()
optimizer.zero_grad()

## also be mindful of changing this. In this example I have resumed from epoch 2
for epoch in range(3,EPOCHS):

    acc_loss = 0.0
    start_time = time.time() *1000

    for batch_idx, (tokens, targets) in enumerate(train_loader):
        
        step = epoch * len(train_loader) + batch_idx + 1
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
                writer.add_scalar("Train Loss",acc_loss,step)
                writer.add_scalar("Tokens/Sec",tokens_per_sec,step)
                writer.add_scalar("Learning Rate", optimizer.lr,step)
            
                if ((batch_idx+1) % VAL_LOG_BATCHES ==0):
                    
                    val_loss = val_loss_tracker.get_val_loss(model)
                    writer.add_scalar("Val Loss",val_loss,step)
                    print(f"Val loss : {val_loss:.4f}")
                    if ((batch_idx+1) % 2*VAL_LOG_BATCHES ==0):
                        log_text(model, step * SEQ_LEN * BATCH_SIZE * TOTAL_AGENTS, val_loss, LOG_FILE)
                    save_checkpoint(
                        model=model,
                        step=step,
                        epoch=epoch,
                        val_loss=val_loss,
                        ckpt_dir="checkpoints"
                    )
                
                writer.flush()

            start_time = time.time() *1000
            acc_loss = 0.0

    if (MASTER_PROCESS):        
        save_checkpoint(
            model=model,
            step=step,
            epoch=epoch,
            val_loss=val_loss,
            ckpt_dir="checkpoints",
            prefix=f"{epoch}"
        )

if IS_DDP:
    dist.destroy_process_group()