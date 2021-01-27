import math
import torch
import torch.nn as nn


class GAT(nn.Module):
    def __init__(self):
        '''
        GAT: Graph Attention Network, ICLR 2018
        https://arxiv.org/pdf/1710.10903.pdf
        '''
        super().__init__()
        self.feat1 = GraphAttn(5, 32, kernel_size=11, stride=1, padding=5, feat_len=25)
        self.acvt1 = nn.Sequential(nn.BatchNorm2d(5), nn.MaxPool2d((1,2)), nn.ReLU())
        self.feat2 = GraphAttn(32, 32, kernel_size=5, stride=1, padding=2, feat_len=12)
        self.acvt2 = nn.Sequential(nn.BatchNorm2d(5), nn.MaxPool2d((1,2)), nn.ReLU())
        self.linear = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(32*12*5, 13))

    def forward(self, x):
        assert x.size(-1)%2 == 0
        x, n = x.split([x.size(-1)//2, x.size(-1)//2], dim=-1)
        x, n = self.feat1(x, x), self.feat1(n, n)
        x, n = self.acvt1(x), self.acvt1(n)
        x, n = self.feat2(x, n), self.feat2(n, x)
        x, n = self.acvt2(x), self.acvt2(n)
        return self.linear(torch.cat([x, n], dim=-1).view(x.size(0), -1))


class GraphAttn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, feat_len, alpha=0.2):
        super().__init__()
        self.out_channels = out_channels
        self.tran = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, padding_mode='circular')
        out_features = out_channels * feat_len
        self.att1 = nn.Linear(out_features, 1, bias=False)
        self.att2 = nn.Linear(out_features, 1, bias=False)
        self.norm = nn.Sequential(nn.Softmax(dim=1))
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, n):
        B, N, C, F = x.shape
        h1 = self.tran(x.view(B*N, C, F)).view(B, N, -1)
        h2 = self.tran(n.view(B*N, C, F)).view(B, N, -1)
        a = self.att1(h1) + self.att2(h2).transpose(-1,-2)
        a = self.leakyrelu(a)
        return (self.norm(a) @ h1).view(B, N, self.out_channels, -1)
