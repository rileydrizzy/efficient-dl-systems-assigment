"""
scaler.py
This module implements a LossScaler class that can be used to scale the loss during training to prevent numerical instability. 
The LossScaler supports both static and dynamic scaling modes. 
In static mode, the scale value is fixed, while in dynamic mode, the scale value is adjusted based on the presence of infinite
 or NaN gradients.
Classes:
- LossScaler: A class that provides methods to scale the loss and update the scale value based on the gradients during training.
Methods:
- scale(loss): Scales the input loss by the current scale value.
- step(optim): Unscales the gradients and performs an optimization step.
    If in dynamic mode, it checks for infinite or NaN gradients and adjusts the scale value accordingly.
- update(): Updates the scale value based on the number of clean steps without infinite or NaN gradients.
    In dynamic mode, if there are 5 consecutive clean steps, the scale value is increased by a factor.

Author: Ladipo Ipadeola
Date: 06/14/2026    

"""


import torch
import torch.nn as nn


class LossScaler():
    def __init__(self, mode="static", scale_value=65536.0, factor=2.0):
        self.found_inf = False
        self.factor = factor
        self.scale_value = scale_value
        self.clean_step_value = 0
        if mode not in ["static", "dynamic"]:
            raise NotImplementedError(
                f"mode {mode} is not implemented, only static and dynamic are supported")
        self.mode = mode

    def scale(self, loss):
        loss = loss * self.scale_value
        return loss

    def step(self, optim):
        for group in optim.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data.div_(self.scale_value)
        if self.mode == "dynamic":
            self.found_inf = any(torch.isinf(param.grad.data).any() or torch.isnan(param.grad.data).any(
            ) for group in optim.param_groups for param in group["params"] if param.grad is not None)
            if self.found_inf:
                self.scale_value = self.scale_value/self.factor
                return None
        optim.step()

    def update(self):
        if self.mode == "static":
            return None
        if self.found_inf:
            self.clean_step_value = 0
        else:
            self.clean_step_value += 1
        if (self.clean_step_value > 0) and (self.clean_step_value % 5 == 0):
            self.scale_value = self.scale_value * self.factor
