import torch
import torch.nn as nn
import math

class CosineSchedulerWithWarmup:


    def __init__(self,lr,named_params,warmup_steps,total_steps,min_lr = None):

        self.start_lr = lr
        self.lr = 0.0
        self._config_optimizer(named_params)
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.step_num = 0

        self._set_lr(0.0)

        if (min_lr == None):
            self.min_lr = self.start_lr*0.1
        else:
            self.min_lr = min_lr

        

    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def _set_lr(self,lr):

        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self.lr = lr

    def step(self):

        self.step_num += 1

        self.set_next_lr()

        self.optimizer.step()
    
    def set_next_lr(self):

        # linear warmup 
        if (self.step_num <= self.warmup_steps):
            self._set_lr(self.start_lr *(self.step_num) / self.warmup_steps)
        
        # Cosine decay
        elif (self.step_num < self.total_steps):
            decay_ratio = (self.step_num - self.warmup_steps) / (self.total_steps - self.warmup_steps)

            self._set_lr(
                self.min_lr + 0.5*(1.0 + math.cos(math.pi*decay_ratio))*(self.start_lr - self.min_lr)
            )
    
    @property
    def param_groups(self):
        return self.optimizer.param_groups
    
    def _config_optimizer(self,named_params):

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


dummy_param_1 = nn.Parameter(torch.randn(10, 10))
dummy_param_2 = nn.Parameter(torch.randn(5, 5))

# Create dummy named_params
named_params = [
    ("linear.weight", dummy_param_1),  # will go to decay group
    ("norm.bias", dummy_param_2)       # will go to no_decay group
]