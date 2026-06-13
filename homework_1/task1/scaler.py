"""
scaler.py
"""


import torch
import torch.nn as nn


class LossScaler(nn.Module):
    def __init__(self, mode, scale=2):
        super().__init__()
        self.scale_num = scale

    def scale(self, loss):
        loss = loss * self.scale_num
        return loss
    
    def step(self, optim):
        """_summary_

        Args:
            optim (_type_): _description_
        """
        optim.step()
        
