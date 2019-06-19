import torch
import torch.nn as nn
import torch.nn.functional as F

# 2d version
# make several dilated convolutional layer with certain dilation rate
''' The dilation rate of the n_th layer is n * dilation_rate  '''
def _make_nDilatedConvs2d(inchannel, outchannel, depth=3, dilation_stride=1):
    units = []
    # make #depth convolutional layrers (parallel)
    for i in range(1, depth+1):
        units.append(nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, dilation=i*dilation_rate, padding=i*dilation_rate))

    return units

class DilatedAttention2d(nn.Module):
    """ docstring for DilatedAttention2d."""
    def __init__(self, inchannel, outchannel):
        super(DilatedAttention2d, self).__init__()
        self.dilated_convs = _make_nDilatedConvs2d(outchannel, outchannel)
        # adapt to the output
        if inchannel != outchannel:
            self.dimExtension = nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=1)
            self.outchannel = outchannel
            self.inchannel = inchannel

        self.softmax = F.softmax

    def forward(x):
        if self.dim_change:
            x = self.dimExtension(x)

        att = torch.zeros(x.size())
        for i in range(len(self.dilated_convs)):
            att += self.dilated_convs[i](x)

        # use of Nonlinearlity
        att = self.softmax(att)
        output = torch.mul(x, att)
        return output

# 3d version
def _make_nDilatedConvs3d(inchannel, outchannel, depth=3, dilation_rate=1):
        units = []
        # make #depth dilated convolutional layers (paralle)
        for i in range(1, depth+1):
            units.append(nn.Conv3d(inchannel, outchannel, kernel_size=3, stride=1, dilation=i*dilation_rate, padding=i*dilation_rate))

        return units

class DilatedAttention3d(nn.Module):
    """docstring for DilatedAttention3d."""
    def __init__(self, inchannel, outchannel):
        super(DilatedAttention3d, self).__init__()
        self.dilated_convs = _make_nDilatedConvs3d(outchannel, outchannel)
        # adapt to the output
        if inchannel != outchannel:
            self.dimExtension = nn.Conv3d(inchannel, outchannel, kernel_size=1, stride=1)
            self.outchannel = outchannel
            self.inchannel = inchannel

        self.softmax = F.softmax

    def forward(self, x):
        if self.outchannel != self.inchannel:
            x = self.dimExtension(x)

        att = torch.zeros(x.size())
        for i in range(len(self.dilated_convs)):
            temp = self.dilated_convs[i](x)
            att += temp

        # use of Nonlinearlity
        att = self.softmax(att)
        #print(x.size())
        output = torch.mul(x, att)
        return output

if __name__ == '__main__':
    model = DilatedAttention3d(2, 4)
    x = torch.randn(2,2,8,8,8)
    y = model(x)
    print(y.size())
