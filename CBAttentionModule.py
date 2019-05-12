import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """docstring for MLP."""
    def __init__(self, in_features, hidden_features, out_features):
        super(MLP, self).__init__()
        self.fc_in = nn.Linear(in_features, hidden_features)
        self.fc_out = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        output = self.fc_in(x)
        output = self.fc_out(output)
        return output

class CBAttenionModule2d(object):
    """docstring for CBAM2d."""
    def __init__(self, arg, inchannel, outchannel, h, w):
        super(CBAttenionModule2d, self).__init__()
        self.conv1 = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=2, padding=1)

        self.spatialMaxPool = nn.MaxPool1d(kernel_size=inchannel, stride=1)
        self.spatialAvgPool = nn.AvgPool1d(kernel_size=inchannel, stride=1)
        self.conv2 = nn.Conv2d(2, outchannel, kernel_size=3, stride=2, padding=1)

        self.channelMaxPool = nn.MaxPool2d(kernel_size=(h, w), stride=1)
        self.channelAvgPool = nn.AvgPool2d(kernel_size=(h, w), stride=1)
        self.mlp = MLP(inchannel, inchannel // 2, inchannel)

        self.sigmoid = torch.sigmoid

    def forward(self, x):
        feature = self.conv1(x)

        chanMaxPool = self.MaxPool2d(x)
        chanAvgPool = self.AvgPool2d(x)
        chanMaxPool = self.mlp(chanMaxPool)
        chanAvgPool = self.mlp(chanAvgPool)
        chanAttention = self.sigmoid(chanAvgPool + chanMaxPool)

        d, n, h, w = x.size()
        x = x.permute(0, 2, 3, 1).contiguous().view(d, h*w, -1)
        spaMaxPool = self.spatialMaxPool(tempx)
        spaMaxPool = spaMaxPool.view(d, 1, h, w)
        spaAvgPool = self.spatialAvgPool(tempx)
        spaAvgPool = spaAvgPool.view(d, 1, h, w)

        spaAttention = torch.concatenate((spaMaxPool, spaAvgPool), dim=1)
        spaAttention = self.conv2(spaAttention)
        spaAttention = self.sigmoid(spaAttention)

        # spatial attention
        feature = torch.mul(spaAttention, feature)
        # channel attention
