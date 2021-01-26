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
        self.feat1 = GraphAttn(5*25, 64)
        self.acvt1 = nn.Sequential(nn.BatchNorm1d(5), nn.ReLU())
        self.feat2 = GraphAttn(64, 32)
        self.acvt2 = nn.Sequential(nn.BatchNorm1d(5), nn.ReLU())
        self.linear = nn.Sequential(nn.Flatten(), nn.Linear(32*2*5, 13))

    def forward(self, x):
        assert x.size(-1)%2 == 0
        x = x.view(x.size(0), x.size(1), -1)
        x, n = x.split([x.size(-1)//2, x.size(-1)//2], dim=-1)
        x, n = self.feat1(x, x), self.feat1(n, n)
        x, n = self.acvt1(x), self.acvt1(n)
        x, n = self.feat2(x, n), self.feat2(n, x)
        x, n = self.acvt2(x), self.acvt2(n)
        return self.linear(torch.cat([x, n], dim=-1))


class GraphAttn(nn.Module):
    def __init__(self, in_features, out_features, alpha=0.2):
        super().__init__()
        self.tran = nn.Linear(in_features, out_features, bias=False)
        self.att1 = nn.Linear(out_features, 1, bias=False)
        self.att2 = nn.Linear(out_features, 1, bias=False)
        self.norm = nn.Sequential(nn.Softmax(dim=1))
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, n):
        h1, h2 = self.tran(x), self.tran(n)
        a = self.att1(h1) + self.att2(h2).transpose(-1,-2)
        a = self.leakyrelu(a)
        return self.norm(a) @ h1
