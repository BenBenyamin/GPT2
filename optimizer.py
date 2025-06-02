import torch
import torch.nn as nn
import math

class CosineSchedulerWithWarmup:
    """
    Implements a learning rate scheduler with linear warmup followed by cosine decay,
    along with AdamW optimizer configuration separating weight decay.
    This is based on GPT3 paper (https://arxiv.org/pdf/2005.14165) appendix B.

    Args:
        lr (float): Starting learning rate.
        named_params (list): List of (name, param) pairs for optimizer configuration.
        warmup_steps (int): Number of steps to linearly increase the learning rate.
        total_steps (int): Total number of training steps.
        min_lr (float, optional): Minimum learning rate after decay. Defaults to 10% of start_lr.
    """
    def __init__(self, lr, named_params, warmup_steps, total_steps, min_lr=None):
        self.start_lr = lr
        self.lr = 0.0
        self._config_optimizer(named_params)
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.step_num = 0

        self._set_lr(0.0)

        self.min_lr = min_lr if min_lr is not None else self.start_lr * 0.1

    def zero_grad(self):
        """Zeroes the gradients of all optimized parameters."""
        self.optimizer.zero_grad()

    def _set_lr(self, lr):
        """
        Sets learning rate across all parameter groups.

        Args:
            lr (float): Learning rate to be set.
        """
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self.lr = lr

    def step(self):
        """
        Performs one optimization step and updates the learning rate accordingly.
        """
        self.step_num += 1
        self.set_next_lr()
        self.optimizer.step()

    def set_next_lr(self):
        """
        Computes and applies the next learning rate based on linear warmup and cosine decay.
        """
        if self.step_num <= self.warmup_steps:
            self._set_lr(self.start_lr * self.step_num / self.warmup_steps)
        elif self.step_num < self.total_steps:
            decay_ratio = (self.step_num - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            self._set_lr(
                self.min_lr + 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) * (self.start_lr - self.min_lr)
            )

    @property
    def param_groups(self):
        """
        Returns parameter groups of the optimizer.

        Returns:
            list: Parameter groups.
        """
        return self.optimizer.param_groups

    def _config_optimizer(self, named_params):
        """
        Configures the AdamW optimizer with separate parameter groups for decay and no-decay.

        Args:
            named_params (list): List of (name, parameter) tuples.
        """
        decay = []
        no_decay = []

        for name, param in named_params:
            if not param.requires_grad:
                continue  # frozen weights
            if name.endswith(".bias") or "norm" in name.lower():
                no_decay.append(param)
            else:
                decay.append(param)

        self.optimizer = torch.optim.AdamW(
            [
                {'params': decay, 'weight_decay': 0.1},
                {'params': no_decay, 'weight_decay': 0.0}
            ],
            lr=self.lr,
            betas=(0.9, 0.95),
            eps=1e-8
        )