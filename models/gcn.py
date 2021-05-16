#!/usr/bin/env python3

import torch
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self):
        '''
        GCN: Graph Convolutional Network, ICLR 2017
        https://arxiv.org/pdf/1609.02907.pdf
        '''
        super().__init__()
        self.feat1 = GraphConv(5, 32, kernel_size=(5,11), stride=1, padding=(0,5), feat_len=25)
        self.acvt1 = nn.Sequential(nn.BatchNorm2d(32), nn.MaxPool2d((1,2)), nn.ReLU())
        self.feat2 = GraphConv(32, 32, kernel_size=(1,5), stride=1, padding=(0,2), feat_len=12)
        self.acvt2 = nn.Sequential(nn.BatchNorm2d(32), nn.MaxPool2d((1,2)), nn.ReLU())
        self.linear = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(32*12, 13))

    def forward(self, x):
        assert x.size(-1)%2 == 0
        x, n = x.split([x.size(-1)//2, x.size(-1)//2], dim=-1)
        x, n = self.feat1(x, x), self.feat1(n, n)
        x, n = self.acvt1(x), self.acvt1(n)
        x, n = self.feat2(x, n), self.feat2(n, x)
        x, n = self.acvt2(x), self.acvt2(n)
        return self.linear(torch.cat([x, n], dim=-1))


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, feat_len, alpha=0.2):
        super().__init__()
        self.tran = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode='circular')
        self.norm = nn.Sequential(nn.Softmax(dim=1))
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, n):
        h1, h2 = self.tran(x), self.tran(n)
        B, N, C, F = h1.shape
        h1 = self.tran(x).view(B, N, C*F)
        h2 = self.tran(n).view(B, N, C*F)
        a = torch.einsum('bnf,bmf->bnm', h1, h2)
        a = self.leakyrelu(a)
        return (self.norm(a) @ h1).view(B, N, C, F)
