#!/usr/bin/env python3

import torch
import torch.nn as nn


class SAGE(nn.Module):
    '''
    GraphSAGE: Inductive Representation Learning on Large Graphs, NIPS 2017
    https://arxiv.org/pdf/1706.02216.pdf
    '''
    def __init__(self, feat_len, num_class, hidden=128, aggr='mean'):
        super().__init__()
        aggrs = {'pool':PoolAggregator, 'mean':MeanAggregator, 'gcn':GCNAggregator}
        Aggregator = aggrs[aggr]
        self.tran1 = Aggregator(feat_len, hidden)
        self.acvt1 = nn.Sequential(nn.BatchNorm1d(1), nn.ReLU())
        self.tran2 = Aggregator(hidden, hidden)
        self.acvt2 = nn.Sequential(nn.BatchNorm1d(1), nn.ReLU())
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(hidden, num_class))

    def forward(self, x, neighbor):
        x, neighbor = self.tran1(x, neighbor)
        x, neighbor = self.acvt1(x), [self.acvt1(n) for n in neighbor]
        x, neighbor = self.tran2(x, neighbor)
        return self.classifier(self.acvt2(x))


class MeanAggregator(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.tranx = nn.Linear(in_features, out_features, False)
        self.trann = nn.Linear(in_features, out_features, False)

    def forward(self, x, neighbor):
        f = torch.cat([n.mean(dim=0, keepdim=True) for n in neighbor])
        x = self.tranx(x) + self.trann(f)
        neighbor = [self.tranx(n) for n in neighbor]
        return x, neighbor


class GCNAggregator(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.tran = nn.Linear(in_features, out_features, False)

    def forward(self, x, neighbor):
        f = torch.cat([n.mean(dim=0, keepdim=True) for n in neighbor])
        x = self.tran(x+f)
        neighbor = [self.tran(n) for n in neighbor]
        return x, neighbor


class PoolAggregator(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.tran = nn.Linear(in_features, out_features, True)

    def forward(self, x, neighbor):
        f = [self.tran(torch.cat([x[i:i+1], n])) for i, n in enumerate(neighbor)]
        x = torch.cat([x.max(dim=0, keepdim=True)[0] for x in f])
        neighbor = [self.tran(n).max(dim=0, keepdim=True)[0] for n in neighbor]
        return x, neighbor