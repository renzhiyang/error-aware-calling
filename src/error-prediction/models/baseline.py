import torch
import torch.nn as nn
import torch.nn.functional as F

class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        

    def forward(self, x):
        return 