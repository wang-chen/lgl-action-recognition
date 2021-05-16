#!/usr/bin/env python3

import torch
import torch.nn as nn


class FGN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat1 = Trans1d(5, 32, kernel_size=(5,11), stride=1, padding=(0,5))
        self.acvt1 = nn.Sequential(nn.BatchNorm2d(32), nn.MaxPool2d((1,2)), nn.ReLU())
        self.feat2 = Trans1d(32, 32, kernel_size=(1,5), stride=1, padding=(0,2))
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


class Trans1d(nn.Module):
    '''
    Temporal Feature Transforming Layer for multi-channel 1D features.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode='circular')

    def forward(self, x, neighbor):
        adj = self.feature_adjacency(x, neighbor)
        return self.transform(x, adj)

    def transform(self, x, adj):
        return self.conv(torch.einsum('bcxy,bncx->bncx', adj, x))

    def feature_adjacency(self, x, y):
        fadj = torch.einsum('bncx,bdcy->bcxy', x, y)
        fadj += fadj.transpose(-2, -1)
        return self.row_normalize(self.sgnroot(fadj))

    def sgnroot(self, x):
        return x.sign()*(x.abs().clamp(min=1e-8).sqrt())

    def row_normalize(self, x):
        x = x / (x.abs().sum(-1, keepdim=True) + 1e-7)
        x[torch.isnan(x)] = 0
        return x
