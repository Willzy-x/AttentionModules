import torch
import torch.nn as nn
import torch.nn.functional as F

# 2d version
class ReverseAttention2d(nn.Module):
    """docstring for ReverseAttention2d."""
    def __init__(self, inchannel, outchannel):
        super(ReverseAttention2d, self).__init__()
        # attention module
        self.attention = nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=1)
        self.softmax = F.softmax
        # adapt to the output
        self.dim_change = nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=1)

    def forward(self, x):
        # attention map
        att = self.softmax(self.attention(x))
        ones = torch.ones(att.size())
        # reverse the attention distribution
        rev_att = ones - att
        # dimension adapting
        x = self.dim_change(x)

        output = torch.mul(rev_att, x)
        return output

# 3d version
class ReverseAttention3d(object):
    """docstring for ReverseAttention3d."""
    def __init__(self, inchannel, outchannel):
        super(ReverseAttention3d, self).__init__()
        self.attention = nn.Conv3d(inchannel, outchannel, kernel_size=1, stride=1)
        self.softmax = F.softmax
        # adapt to the output
        self.dim_change = nn.Conv3d(inchannel, outchannel, kernel_size=1, stride=1)

    def forward(self, x):
        # attention map
        att = self.softmax(self.attention(x))
        ones = torch.ones(att.size())
        # reverse the attention distribution
        rev_att = ones - att
        # dimension adapting
        x = self.dim_change(x)

        output = torch.mul(rev_att, x)
        return output
