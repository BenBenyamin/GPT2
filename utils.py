import torch
from torch.amp import autocast
from loss import GPT2Loss
import tiktoken
from torch.nn import functional as F
import os

class ValLossTracker:
    """
    Tracks validation loss over a rolling window of batches.

    Args:
        val_loader (iterable): Iterable validation data loader (list of (tokens, targets) pairs).
        num_of_batches (int): Number of batches to evaluate per call.
    """
    def __init__(self, val_loader, num_of_batches):
        self.val_loader = list(val_loader)
        self.num_of_batches = num_of_batches
        self.rolling_idx = 0
        self.loss = GPT2Loss()

    @torch.no_grad()
    def get_val_loss(self, model):
        """
        Computes the average validation loss over a fixed number of batches.

        Args:
            model (nn.Module): The model to evaluate.

        Returns:
            float: Average validation loss.
        """
        model.eval()
        device = next(model.parameters()).device
        acc_loss = 0.0

        for batch_idx in range(self.rolling_idx, self.rolling_idx + self.num_of_batches):
            tokens, targets = self.val_loader[batch_idx]
            tokens, targets = tokens.to(device), targets.to(device)

            with autocast(device_type='cuda'):
                logits = model(tokens)
                acc_loss += self.loss(logits, targets).item()

        self.rolling_idx += 1
        if self.rolling_idx + self.num_of_batches > len(self.val_loader):
            self.rolling_idx = 0

        model.train()
        return acc_loss / self.num_of_batches

def log_text(model, tokens_processed, val_loss, out_file):
    """
    Generates and logs text samples from the model, along with validation loss and tokens processed.

    Args:
        model (nn.Module): The trained model for inference.
        tokens_processed (int): Number of tokens processed so far.
        val_loss (float): Current validation loss.
        out_file (str): Path to the output log file.
    """
    model.eval()
    enc = tiktoken.get_encoding('gpt2')

    tokens = [15496, 11, 314, 1101, 257, 3303, 2746, 11]  # "Hello, I'm a language model,"
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).repeat(5, 1).to('cuda')

    SEQ_LEN = 200
    x = tokens

    while x.size(1) < SEQ_LEN:
        with torch.no_grad():
            logits = model(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            x = torch.cat((x, xcol), dim=1)

    with open(out_file, "a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Validation Loss: {val_loss:.4f}\n")
        f.write(f"Tokens Processed: {tokens_processed}\n")
        f.write("Model Outputs:\n\n")

        for i in range(5):
            decoded = enc.decode(x[i, :SEQ_LEN].tolist())
            f.write(f"[Sample {i+1}]\n")
            f.write(decoded.strip() + "\n\n")
        f.write("=" * 80 + "\n\n")

    model.train()

def save_checkpoint(model, step, epoch, val_loss, ckpt_dir="checkpoints", prefix="gpt2"):
    """
    Saves model checkpoint to disk.

    Args:
        model (nn.Module): Model or DDP-wrapped model.
        step (int): Current training step.
        epoch (int): Current training epoch.
        val_loss (float): Latest validation loss.
        ckpt_dir (str): Directory to save checkpoint.
        prefix (str): Prefix name for the checkpoint file.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model

    ckpt_path = os.path.join(
        ckpt_dir,
        f"{prefix}_checkpoint.pt"
    )

    torch.save({
        "model_state_dict": model_to_save.state_dict(),
        "epoch": epoch,
        "step": step,
        "val_loss": val_loss,
    }, ckpt_path)

    print(f"[Checkpoint saved to] {ckpt_path}")
