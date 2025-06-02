import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.amp import autocast

class GPT2Loss(nn.Module):
    """
    Cross-entropy loss for GPT-2 models, ignoring the padding token (ID 50256).

    This module reshapes the input and target tensors to apply cross-entropy loss
    across the vocabulary dimension and ignores padding tokens during loss computation.
    """
    def __init__(self):
        """
        Initializes the GPT2Loss module with no additional parameters.
        """
        super().__init__()

    def forward(self, logits, targets):
        """
        Computes cross-entropy loss between logits and targets.

        Args:
            logits (Tensor): Predicted unnormalized log probabilities (B, T, vocab_size).
            targets (Tensor): Ground truth token indices (B, T).

        Returns:
            Tensor: Scalar loss value.
        """
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=50256  # Ignore padding token
        )
        return loss
