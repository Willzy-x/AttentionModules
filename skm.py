import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce

class SKM2d(nn.Module):
    """docstring for SKM2d"""
    def __init__(self, input_channel, out_channel, kernels=[3,5], reduction_rate=16):
        super(SKM2d, self).__init__()
        self.convs = []
        self.M = len(kernels)
        self.out_channel = out_channel
        self.eff_d = max(out_channel/reduction_rate, 32)
        for i in kernels:
            self.convs.append(nn.Sequential(nn.Conv2d(input_channel, out_channel, kernel_size=i, padding=i//2, groups=input_channel),
                                            nn.BatchNorm2d(out_channel),
                                            nn.ReLU(inplace=True)))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(nn.Linear(out_channel, self.eff_d),
                                 nn.ReLU(inplace=True))
        self.fc2 = nn.Linear(self.eff_d, out_channel*self.M)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        bz = x.size(0)
        outs = []
        for layer in self.convs:
            outs.append(layer(x))

        U = reduce(lambda x,y:x+y,outs)
        z = self.fc1(self.gap(U).squeeze(3).squeeze(2))

        a_b = self.fc2(z)
        a_b = a_b.reshape(bz, self.M, self.out_channel, -1)
        a_b = self.softmax(a_b)
        a_b=list(a_b.chunk(self.M,dim=1))#split to a and b
        a_b=list(map(lambda x:x.reshape(bz,self.out_channel,1,1),a_b))
        V=list(map(lambda x,y:x*y,outs,a_b))
        V=reduce(lambda x,y:x+y,V)
        return V

class SKM3d(nn.Module):

    def __init__(self, input_channel, out_channel, kernels=[3,5], reduction_rate=16):
        super(SKM3d, self).__init__()
        self.convs = []
        self.M = len(kernels)
        self.out_channel = out_channel
        self.eff_d = max(out_channel/reduction_rate, 32)
        for i in kernels:
            self.convs.append(nn.Sequential(nn.Conv3d(input_channel, out_channel, kernel_size=i, padding=i//2, groups=input_channel),
                                            nn.BatchNorm3d(out_channel),
                                            nn.ReLU(inplace=True)))
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Sequential(nn.Linear(out_channel, self.eff_d),
                                 nn.ReLU(inplace=True))
        self.fc2 = nn.Linear(self.eff_d, out_channel*self.M)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        bz = x.size(0)
        outs = []
        for layer in self.convs:
            outs.append(layer(x))

        U = reduce(lambda x,y:x+y,outs)
        z = self.fc1(self.gap(U).squeeze(4).squeeze(3).squeeze(2))

        a_b = self.fc2(z)
        a_b = a_b.reshape(bz, self.M, self.out_channel, -1)
        a_b = self.softmax(a_b)
        a_b=list(a_b.chunk(self.M,dim=1))#split to a and b
        a_b=list(map(lambda x:x.reshape(bz,self.out_channel,1,1,1),a_b))
        V=list(map(lambda x,y:x*y,outs,a_b))
        V=reduce(lambda x,y:x+y,V)
        return V


if __name__ == '__main__':
    sk = SKM3d(16,16,reduction_rate=2)
    x = torch.randn([2,16,8,8,8])
    y = sk(x)
    print(y.shape)
