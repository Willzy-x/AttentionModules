import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedAttention2d(nn.Module):
    """ docstring for DilatedAttention2d."""
    def __init__(self, inchannel, outchannel):
        super(DilatedAttention2d, self).__init__()
        self.convd1 = nn.Conv2d(inchannel, outchannel, stride=1, kernel_size=3, dilation=1)
