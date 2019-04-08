import torch
import torch.nn as nn
import torch.nn.functional as F

''' return #depth residual units '''
def _make_nResidualUnits(inchannel, outchannel, depth):
    units = []
    # handle unequal channels
    if inchannel != outchannel:
        for i in range(depth):
            if i == depth // 2:
                units.append(ResidualUnit2d(inchannel, outchannel))
            else:
                units.append(ResidualUnit2d(inchannel, inchannel))

    else:
        for _ in range(depth):
            units.append(ResidualUnit2d(inchannel, outchannel))

    return nn.Sequential(*units)

# Basic ResidualBlock
class ResidualUnit2d(nn.Module):
    """docstring for ResidualUnit2d."""
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualUnit2d, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)

        out = F.relu(out)

        return out

# ResidualAttentionModule
class ResidualAttention2d(nn.Module):
    """ docstring for ResidualAttention2d.
        Hyper parameter:
        p: number of pre-processing Residual Unit before splitting into trunk branch and mask branch
        t: number of Residual Unit in the trunk branch
        r: number of Residual Unit between adjacent pooling layer in the mask branch
    """
    def __init__(self, inchannel, outchannel, p=1, t=2, r=1):
        super(ResidualAttention2d, self).__init__()
        self.preBranch = _make_nResidualUnits(inchannel, inchannel, p)

        self.trunkBranch = _make_nResidualUnits(inchannel, outchannel, t)

        self.softMaskBranch = nn.Sequential(
            # Down sampling -- 1
            nn.MaxPool2d(kernel_size=3, stride=2),
            _make_nResidualUnits(inchannel, inchannel, r),
            # Down sampling -- 2
            nn.MaxPool2d(kernel_size=3, stride=2),
            _make_nResidualUnits(inchannel, inchannel, 2*r),
            # interpolation(scale_factor=2, mode='bilinear')
            # Up sampling -- 1
            nn.UpsamplingBilinear2d(scale_factor=2),
            _make_nResidualUnits(inchannel, outchannel, r),
            # Up sampling -- 2
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(outchannel, outchannel, kernel_size=1, stride=1),
            nn.Conv2d(outchannel, outchannel, kernel_size=1, stride=1),
        )

        self.sigmoid = torch.sigmoid

    def forward(self, x):
        p_output = self.preBranch(x)

        t_output = self.trunkBranch(p_output)

        s_output = self.trunkBranch(p_output)
        s_output = self.sigmoid(s_output)

        output = torch.mul(t_output, s_output)
        output += t_output

        return output
