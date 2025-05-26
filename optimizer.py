import torch
import math

class CosineSchedulerWithWarmup:

    """
        start_lr (float): The learning rate to reach at the end of warm-up.
        lr (float): The current learning rate.
        warmup_steps (int): Number of steps over which to linearly increase LR.
        total_steps (int): Total number of scheduling steps (warm-up + decay).
        min_lr (float): The floor learning rate after cosine decay.
        step_num (int): Counter for how many `step()` calls have been made.
    """
    def __init__(self,lr,params,warmup_steps,total_steps,min_lr = None):

        self.optimizer = torch.optim.AdamW(params=params,lr=lr)
        self.start_lr = lr
        self.lr = 0.0
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
