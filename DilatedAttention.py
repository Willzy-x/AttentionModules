import torch
import torch.nn as nn
import torch.nn.functional as F

# 2d version
# make several dilated convolutional layer with certain dilation rate
''' The dilation rate of the n_th layer is n * dilation_rate  '''
def _make_nDilatedConvs2d(inchannel, outchannel, depth=3, dilation_stride=1):
    units = []
    # make #depth convolutional layrers (parallel)
    for i in depth:
        units.append(nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, dilation=i*dilation_rate, padding=i*dilation_rate))

    return units

class DilatedAttention2d(nn.Module):
    """ docstring for DilatedAttention2d."""
    def __init__(self, inchannel, outchannel):
        super(DilatedAttention2d, self).__init__()
        self.dilated_convs = _make_nDilatedConvs2d(inchannel, outchannel)
        self.dim_change = False
        # adapt to the output
        if inchannel != outchannel:
            self.dimExtension = nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=1)
            self.dim_change = True

        self.softmax = F.softmax

    def forward(x):
        att = torch.zeros(x.size())

        for i in len(self.dilated_convs):
            att += self.dilated_convs[i](x)

        if self.dim_change:
            x = self.dimExtension(x)
        # use of Nonlinearlity
        att = self.softmax(att)
        output = torch.mul(x, att)
        return output

# 3d version
def _make_nDilatedConvs3d(inchannel, outchannel, depth=3, dilation_rate=1):
        units = []
        # make #depth dilated convolutional layers (paralle)
        for i in depth:
            units.append(nn.Conv3d(inchannel, outchannel, kernel_size=3, stride=1, dilation=i*dilation_rate, padding=i*dilation_rate))

        return units

class DilatedAttention3d(nn.Module):
    """docstring for DilatedAttention3d."""
    def __init__(self, inchannel, outchannel):
        super(DilatedAttention3d, self).__init__()
        self.dilated_convs = _make_nDilatedConvs3d(inchannel, outchannel)
        self.dim_change = False
        # adapt to the output
        if inchannel != outchannel:
            self.dimExtension = nn.Conv3d(inchannel, outchannel, kernel_size=1, stride=1)
            sefl,dim_change = True

        self.softmax = F.softmax

    def forward(x):
        att = torch.zeros(x.size())

        for i in len(self.dilated_convs)
            att += self.dilated_convs[i](x)

        if self.dim_change:
            x = self.dimExtension(x)
        # use of Nonlinearlity
        att = self.softmax(att)
        output = torch.mul(x, att)
        return output


        return
