# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import time

# Basic biLSTM Module 2d image <-> 2 biLSTMs
# 2d version
class Renet2d(nn.Module):
    """docstring for Renet2d."""
    def __init__(self, size, in_channel, out_channel):
        super(Renet2d, self).__init__()
        self.size = size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.vertical = nn.LSTM(input_size=in_channel, hidden_size=256,
                                batch_first=True, bidirectional=True) # each row
        self.horizontal = nn.LSTM(input_size=512, hidden_size=256,
                                  batch_first=True, bidirectional=True) # each column
        self.conv = nn.Conv2d(512, out_channel, 1)

    def forward(self, *input):
        x = input[0]
        _, _, h, w = x.size()
        # vertical process
        temp = []
        x = torch.transpose(x, 1, 3) # batch, width, height, in_channel
        for i in range(h): # size = height
            h, _ = self.vertical(x[:, :, i, :]) # height dimension
            temp.append(h) # batch, width, 512
        x = torch.stack(temp, dim=2) # batch, width, height, 512 = hidden_size*bidirectional
        # horizontal process
        temp = []
        for i in range(w): # size = width
            h, _ = self.horizontal(x[:, i, :, :])
            temp.append(h) # batch, width, 512
        x = torch.stack(temp, dim=3) # batch, height, 512, width
        x = torch.transpose(x, 1, 2) # batch, 512, height, width
        x = self.conv(x)
        return x

# 2d version
class PicanetG2d(nn.Module):
    """docstring for PicanetG2d."""
    def __init__(self, size, in_channel):
        super(PicanetG2d, self).__init__()
        self.renet = Renet2d(size, in_channel, 100)
        self.in_channel = in_channel

    def forward(self, *input):
        x = input[0]
        size = x.size()
        kernel = F.softmax(self.renet(x), 1)
        # unfold parameter is determined by the input size
        x = F.unfold(x, [10, 10], dilation=[3, 3])
        # output batch, channel*10*10, L= (size0 - 3x(10-1)) x (size1 -3x(10-1))
        # original size: 28, 28, 28, 56, 112, 224
        x = x.reshape(size[0], size[1],  10 * 10) # ?? 10 * 10
        kernel = kernel.reshape(size[0], 100, -1)
        x = torch.matmul(x, kernel)
        x = x.reshape(size[0], size[1], size[2], size[3])
        return x

class PicanetL2d(nn.Module):
    def __init__(self, in_channel):
        super(PicanetL2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 128, kernel_size=7, dilation=2, padding=6)
        self.conv2 = nn.Conv2d(128, 49, kernel_size=1)
        # 49 is intended to match the kernel_size in unfold

    def forward(self, *input):
        x = input[0]
        size = x.size()
        kernel = self.conv1(x)
        kernel = self.conv2(kernel)
        kernel = F.softmax(kernel, 1)
        kernel = kernel.reshape(size[0], 1, size[2] * size[3], 7 * 7)
        # channel at last in order
        x = F.unfold(x, kernel_size=[7, 7], dilation=[2, 2], padding=6)
        # output: batch, channel*7*7, L=(size0+2x6-2x(7-1)) * (size1+2x6-2x(7-1))
        # L = size0 x size1
        x = x.reshape(size[0], size[1], size[2] * size[3], -1) # 7*7
        # print(x.shape, kernel.shape)
        x = torch.mul(x, kernel)
        x = torch.sum(x, dim=3)
        x = x.reshape(size[0], size[1], size[2], size[3])
        return x

# Basic biLSTM Module 3d image <-> 3 biLSTMs ?
# 3d version
class Renet3d(nn.Module):
    """docstring for Renet3d."""
    def __init__(self, size, in_channel, out_channel):
        super(Renet3d, self).__init__()
        self.size = size
        self.in_channel = in_channel
        self.out_channel = out_channel

        '''
        self.D_axis = nn.LSTM(input_size=in_channel, hidden_size=256, batch_first=True,
                              bidirectional=True) # each D
        self.H_axis = nn.LSTM(input_size=512, hidden_size=256, batch_first=True,
                              bidirectional=True) # each row
        self.W_axis = nn.LSTM(input_size=512, hidden_size=256, batch_first=True,
                              bidirectional=True) # eachh column
        '''

        # composed of 3 Renet2ds
        self.Renet2d_D = Renet2d(size, in_channel, in_channel)
        self.Renet2d_W = Renet2d(size, in_channel, in_channel)
        self.Renet2d_H = Renet2d(size, in_channel, in_channel)
        self.conv = nn.Conv3d(in_channel, out_channel, kernel_size=1)

    def forward(self, *input):
        x = input[0]
        _, _, d, h, w = x.size()
        # D_axis - 2d images
        temp = []
        for i in range(d):
            t = self.Renet2d_D(x[:, :, i, :, :])
            temp.append(t) # batch, out_channel, h, w
        x = torch.stack(temp, dim=2) # batch, out_channel, d, h, w
        # W_axis - 2d images
        temp = []
        for i in range(w):
            t = self.Renet2d_D(x[:, :, :, i, :])
            temp.append(t) # batch, out_channel, d, w
        x = torch.stack(temp, dim=3) # batch, out_channel, d, h, w
        # H_axis - 2d images
        temp = []
        for i in range(h):
            t = self.Renet2d_D(x[:, :, :, :, i])
            temp.append(t) # batch, out_channel, d, h
        x = torch.stack(temp, dim=4) # batch, out_channel, d, h, w

        x = self.conv(x)
        return x


# 3d version
class PicanetG3d(nn.Module):
    """docstring for PicanetG3d."""
    def __init__(self, size, in_channel):
        super(PicanetG3d, self).__init__()
        self.PicaG2d = PicanetG2d(size, in_channel)
        self.in_channel = in_channel

    def forward(self, *input):
        x = input[0]
        # computing on slice axis
        _, _, d, h, w = x.size()
        temp = []
        for i in range(d):
            temp.append(self.PicaG2d(x[:, :, i, :, :]))
        x = torch.stack(temp, dim=2)
        return x

class PicanetL3d(nn.Module):
    """docstring for PicanetL3d."""

    def __init__(self, in_channel):
        super(PicanetL3d, self).__init__()
        '''
        self.conv1 = nn.Conv3d(in_channel, 128, kernel_size=7, dilation=2, padding=0)
        self.conv2 = nn.Conv3d(128, 49, kernel_size=1)
        '''
        self.picaL2d = PicanetL2d(in_channel)
        self.in_channel = in_channel

    def forward(self, *input):

        x = input[0]
        size = x.size()
        # attention distribution
        _, _, d, h, w = x.size()
        temp = []
        for i in range(d):
            temp.append(self.PicaL2d(x[:, :, i, :, :]))
        x = torch.stack(temp, dim=2)
        '''
        kernel = F.softmax(self.conv2(self.conv1(x)), dim=1)
        # -------------------------------------------------

        # -------------------------------------------------
        x = torch.mul(x, kernel)
        x = torch.sum(x, dim=?)
        x = x.reshape(size[0], size[1], size[2], size[3], size[4])
        '''
        return x

if __name__ == '__main__':
    test_x = torch.randn([1, 4, 8, 28, 28])
    pica = PicanetG3d([8, 28, 28], 4)
    out = pica(test_x)
    #r3 = Renet3d([8, 8, 8], 4, 8)

    #out1 = r3(test_x)
    print(out.size())
