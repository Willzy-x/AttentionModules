import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleAttention2d(nn.Module):
    """docstring for DoubleAttention."""
    def __init__(self, inchannel, outchannel, reduced_dim):
        super(DoubleAttention2d, self).__init__()

        ''' First Attention Step: Feature Gathering '''
        self.dimReduction = nn.Conv2d(inchannel, reduced_dim, kernel_size=1, stride=1)
        self.attention1 = nn.Conv2d(inchannel, inchannel, kernel_size=1, stride=1)
        ''' Second Attention Step: Feature Distribution '''
        self.attention2 = nn.Conv2d(inchannel, inchannel, kernel_size=1, stride=1)
        self.dimExtension = nn.Conv2d(reduced_dim, outchannel, kernel_size=1, stride=1)

        # Nonlinearlity
        self.softmax = F.softmax

    def forward(self, x):
        d, c_input, h, w = x.size()

        # dimReduction
        tmpA = self.dimReduction(x)
        _, c_red, _, _ = tmpA.size()
        tmpB = self.attention1(x)

        # A.shape: m x dhw
        # B.shape: n x dhw -> transpose: dhw x n
        tmpA = (tmpA.permute(1, 0, 2, 3)).contiguous().view(c_red, d*h*w)
        tmpB = (tmpB.permute(1, 0, 2, 3)).contiguous().view(c_input, d*h*w)
        tmpB = self.softmax(tmpB, dim=0).permute(1, 0)

        # output1: m x n
        output1 = torch.matmul(tmpA, tmpB)

        # V.shape: (n x dhw)
        tmpV = self.attention2(x)
        tmpV = (tmpV.permute(1, 0, 2, 3)).contiguous().view(c_input, d*h*w)
        tmpV = self.softmax(tmpV, dim=0)
        output = torch.matmul(output1, tmpV)

        # dimExtension
        output = output.contiguous().view(c_red, d, h, w).permute(1, 0, 2, 3)
        output = self.dimExtension(output)

        return output

def getDoubleAttention(inchannel, outchannel, reduced_dim):
    return DoubleAttention2d(inchannel, reduced_dim, outchannel)
