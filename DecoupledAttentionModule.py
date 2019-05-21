import torch
import torch.nn as nn
import torch.nn.functional as F

# 2d version
class DecoupledAttention2d(nn.Module):
    """ docstring for DecoupledAttention2d.
        increase drop rate to get more extensive attention
    """
    def __init__(self, inchannel, outchannel, drop_rate=0.5):
        super(DecoupledAttention2d, self).__init__()
        ''' Extensive part '''
        self.drop1 = nn.Dropout2d(drop_rate)
        self.exConv = nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=1)
        self.drop2 = nn.Dropout2d(drop_rate)
        # non-linear activation function
        self.lrelu = nn.LeakyReLU()
        self.norm = F.normalize

        ''' Discriminative part '''
        self.disConv = nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=1)

    def forward(self, x):
        # outputE:
        outputE = self.drop1(x)
        outputE = self.exConv(outputE)
        outputE = self.drop2(outputE)
        outputE = self.lrelu(outputE)
        outputE = self.norm(outputE)
        # d, c, h, w = outputE.size()

        # outputD:
        outputD = self.disConv(x)

        # element-wise product
        output = torch.mul(outputE, outputD)

        return output

# 3d version
class DecoupledAttention3d(nn.Module):
    """ docstring for DecoupledAttention2d.
        increase drop rate to get more extensive attention
    """
    def __init__(self, inchannel, outchannel, drop_rate=0.5):
        super(DecoupledAttention3d, self).__init__()
        ''' Extensive part '''
        self.drop1 = nn.Dropout3d(drop_rate)
        self.exConv = nn.Conv3d(inchannel, outchannel, kernel_size=1, stride=1)
        self.drop2 = nn.Dropout3d(drop_rate)
        # non-linear activation function
        self.lrelu = nn.LeakyReLU()
        self.norm = F.normalize

        ''' Discriminative part '''
        self.disConv = nn.Conv3d(inchannel, outchannel, kernel_size=1, stride=1)

    def forward(self, x):
        # outputE:
        outputE = self.drop1(x)
        outputE = self.exConv(outputE)
        outputE = self.drop2(outputE)
        outputE = self.lrelu(outputE)
        outputE = self.norm(outputE)
        # d, c, h, w = outputE.size()

        # outputD:
        outputD = self.disConv(x)

        # element-wise product
        output = torch.mul(outputE, outputD)

        return output
