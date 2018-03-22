"""
Implementation of EG plus/minus optimizer in PyTorch

Author: Keller Jordan
"""

import torch
from torch.optim.optimizer import Optimizer, required

class EGPM(Optimizer):

    def __init__(self, params, lr=required, u_scaling=100):        
        defaults = dict(lr=lr, u_scaling=u_scaling)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            u = group['u_scaling']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                
                param_state = self.state[p]
                # initialize positive and negative weights
                if 'egpos' not in param_state:
                    param_state['egpos'] = (u + p.data.clone()) / 2
                    param_state['egneg'] = (u - p.data.clone()) / 2
                
                egpos = param_state['egpos']
                egneg = param_state['egneg']
                
                rpos = torch.exp(-lr * d_p)
                rneg = torch.exp(lr * d_p)
                updatepos = rpos * egpos
                updateneg = rneg * egneg
                
                # constant to normalize sum of weights to 1
                Z = torch.mean(updatepos + updateneg)
                
                egpos = param_state['egpos'] = u * updatepos / Z
                egneg = param_state['egneg'] = u * updateneg / Z

                p.data = egpos - egneg
    
        return loss
