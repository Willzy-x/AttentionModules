import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossbarAttention2d(nn.Module):
    """docstring for CrossbarAttention2d."""

    def __init__(self, in_channel, out_channel, height, width):
        super(CrossbarAttention2d, self).__init__()
        self.convH = nn.Conv2d(in_channel, 1, kernel_size=(1, width), stride=1)
        self.convW = nn.Conv2d(in_channel, 1, kernel_size=(height, 1), stride=1)

        self.fc = nn.Linear(height+width, out_channel)
        # self.fcW = nn.Linear(width, out_channel)

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1)


    def forward(self, x):

        _, _, h, w = x.size()
        a_h = self.convH(x)
        size_h = a_h.size()
        # batch, 1, height, 1
        a_h = a_h.view(size_h[0], -1).contiguous()

        a_w = self.convW(x)
        size_w = a_w.size()
        # batch, 1, 1, width
        a_w = a_w.view(size_w[0], -1).contiguous()
        x = self.conv1(x)
        #print(a_w.size())
        spatial_att = torch.einsum('bi,bj->bij', [a_h, a_w])
        spatial_att = F.softmax(spatial_att, dim=0)
        x = torch.einsum('bij,bcij->bcij', [spatial_att, x])
        #print(a_h.size(), a_w.size())
        # batch, out_channel
        chan_att = self.fc(torch.cat([a_h, a_w], dim=1))
        chan_att = F.softmax(chan_att, dim=0)
        x = torch.einsum('bc,bcij->bcij', [chan_att, x])

        return x

class CrossbarAttention3d(nn.Module):
    """docstring for CrossbarAttention3d."""

    def __init__(self, in_channel, out_channel, height, width):
        super(CrossbarAttention3d, self).__init__()
        self.height = height
        self.width = width
        self.convH = nn.Conv2d(in_channel, 1, kernel_size=(1, width), stride=1)
        self.convW = nn.Conv2d(in_channel, 1, kernel_size=(height, 1), stride=1)

    def forward(self, x):
        b, c, slice, _, _ = x.size()
        x = x.view(b*slice, c, self.height, self.width).contiguous()
        a_h = self.convH(x)
        size_h = a_h.size()
        a_h = a_h.view(size_h[0], -1).contiguous()

        a_w = self.convW(x)
        size_w = a_w.size()
        a_w = a_w.view(size_w[0], -1).contiguous()

        spatial_att = torch.einsum('bi,bj->bij', [a_h, a_w])
        spatial_att = F.softmax(spatial_att, dim=0)
        x = torch.einsum('bij,bcij->bcij', [spatial_att, x])

        x = x.view(b, c, slice, self.height, self.width).contiguous()
        '''
        a_h = []
        a_w = []
        for i in range(slice):
            a_h.append(self.conv)
        '''
        return x


if __name__ == '__main__':
    x = torch.randn(2,2,3,4,8)
    cba = CrossbarAttention3d(2, 4, 4, 8)
    y = cba(x)
    print(y.size())
