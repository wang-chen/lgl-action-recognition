import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(5, 32, kernel_size=(5,10), stride=2, padding=0)
        self.acvt1 = nn.Sequential(nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(1,10), stride=2, padding=0)
        self.acvt2 = nn.Sequential(nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.linear = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(32 * 6, 13))

    def forward(self, x):
        x = self.conv1(x)
        x = self.acvt1(x)
        x = self.conv2(x)
        x = self.acvt2(x)
        return self.linear(x)


class FGN(nn.Module):
    def __init__(self):
        super().__init__()
        self.tran1 = Trans1d(5, 32, kernel_size=(5,10), stride=1, padding=0)
        self.acvt1 = nn.Sequential(nn.BatchNorm2d(32), nn.ReLU())
        self.tran2 = Trans1d(32, 32, kernel_size=(1,10), stride=2, padding=0)
        self.acvt2 = nn.Sequential(nn.BatchNorm2d(32), nn.ReLU())
        self.linear = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(32*8, 13))

    def forward(self, x):
        assert x.size(-1)%2 == 0
        x, n = x.split([x.size(-1)//2, x.size(-1)//2], dim=-1)
        x, n = self.tran1(x, x), self.tran1(n, n)
        x, n = self.acvt1(x), self.acvt1(n)
        x, n = self.tran2(x, n), self.tran2(n, x)
        x, n = self.acvt2(x), self.acvt2(n)
        x = torch.cat([x, n], dim=-2)
        return self.linear(x)


class Trans1d(nn.Module):
    '''
    Temporal Feature Transforming Layer for multi-channel 1D features.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x, neighbor):
        adj = self.feature_adjacency(x, neighbor)
        return self.transform(x, adj)

    def transform(self, x, adj):
        return self.conv(0.9*x + 0.1*torch.einsum('bcxy,bncx->bncx', adj, x))

    def feature_adjacency(self, x, y):
        fadj = torch.einsum('bncx,bdcy->bcxy', x, y)
        fadj += fadj.transpose(-2, -1)
        return self.row_normalize(self.sgnroot(fadj))

    def sgnroot(self, x):
        return x.sign()*(x.abs().sqrt())

    def row_normalize(self, x):
        x = x / (x.abs().sum(-1, keepdim=True) + 1e-7)
        x[torch.isnan(x)] = 0
        return x
