import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(5, 32, kernel_size=(5,10), stride=2, padding=0)
        self.acvt1 = nn.Sequential(nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(1,10), stride=2, padding=0)
        self.acvt2 = nn.Sequential(nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.linear = nn.Sequential(nn.Flatten(), nn.Linear(32 * 6, 13))

    def forward(self, x):
        x = self.conv1(x)
        x = self.acvt1(x)
        x = self.conv2(x)
        x = self.acvt2(x)
        return self.linear(x)


class FGN(nn.Module):
    def __init__(self):
        super().__init__()
        self.tran1 = Trans1d(5, 32, kernel_size=(5,10), stride=2, padding=0)
        self.acvt1 = nn.Sequential(nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.tran2 = Trans1d(32, 32, kernel_size=(1,10), stride=2, padding=0)
        self.acvt2 = nn.Sequential(nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.linear = nn.Sequential(nn.Flatten(), nn.Linear(32 * 6, 13))

    def forward(self, x):
        x, y = self.tran1(x, x)
        x, y = self.acvt1(x), self.acvt1(y)
        x, y = self.tran2(x, y)
        x, y = self.acvt2(x), self.acvt2(y)
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
        x = self.transform(x, adj)
        neighbor = self.transform(neighbor, adj)
        return x, neighbor

    def transform(self, x, adj):
        return self.conv(torch.einsum('bcxy,bncx->bncx', adj, x))

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
