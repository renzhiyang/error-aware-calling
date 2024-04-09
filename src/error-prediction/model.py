import torch
import torch.nn as nn

class ErrorPrediction(nn.Module):
    def __init__(self):
        super(ErrorPrediction, self).__init__()
        
    def forward(self):
        return