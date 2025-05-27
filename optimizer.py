import torch
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

        # linear warmup 
        if (self.step_num <= self.warmup_steps):
            self._set_lr(self.start_lr *(self.step_num) / self.warmup_steps)
        
        # Cosine decay
        elif (self.step_num < self.total_steps):
            decay_ratio = (self.step_num - self.warmup_steps) / (self.total_steps - self.warmup_steps)

            self._set_lr(
                self.min_lr + 0.5*(1.0 + math.cos(math.pi*decay_ratio))*(self.start_lr - self.min_lr)
            )
        
        self.optimizer.step()
    
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
       

        
### TEST ###
# import numpy as np
# import matplotlib.pyplot as plt



# dummy_param = [torch.nn.Parameter(torch.zeros(1))]

# ###Optimizer

# optimizer = CosineSchedulerWithWarmup(
#     lr = 6e-4,
#     params = dummy_param,
#     warmup_steps=10,
#     total_steps=50
# )

# # Generate LR values
# steps = np.arange(optimizer.total_steps)
# lrs = []

# for step in range(optimizer.total_steps):

#     lrs.append(optimizer.lr)
#     optimizer.step()

# lrs = np.array(lrs)

# # Plot
# plt.figure()
# plt.plot(steps, lrs)
# plt.xlabel("Step")
# plt.ylabel("Learning Rate")
# plt.title("Warm-up + Cosine Decay Schedule")
# plt.show()
