import os
import numpy as np
from inspect import isfunction
from copy import deepcopy

import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms


def default(val, d):
    """a help function that sets val to d if val is not specified"""
    if val is not None:
        return val
    return d() if isfunction(d) else d

def save_checkpoint(state, is_final, file_folder, file_name="checkpoint.pth.tar"):
    """save checkpoint to file"""
    if not os.path.exists(file_folder):
        os.mkdir(file_folder)
    torch.save(state, os.path.join(file_folder, file_name))
    if is_final:
        # skip the optimization / scheduler state
        state.pop("optimizer", None)
        torch.save(state, os.path.join(file_folder, "model_final.pth.tar"))

class ModelEMA(torch.nn.Module):
    """
    maintain moving averages of the trained parameters
    """

    def __init__(self, model, decay=0.999, device=None):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        # perform ema on different device from model if set
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(
                self.module.state_dict().values(),
                model.state_dict().values()
            ):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(
            model,
            update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m
        )

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

class AverageMeter(object):
    """
    Compute and store the average and current value.
    Used to compute dataset stats from mini-batches
    """

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0.0

    def initialize(self, val, n):
        self.val = val
        self.avg = val
        self.sum = val * n
        self.count = n
        self.initialized = True

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
