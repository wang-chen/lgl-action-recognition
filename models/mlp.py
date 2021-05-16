#!/usr/bin/env python3

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat1 = nn.Sequential(nn.Flatten(), nn.Linear(50*5*5, 32*5*5), nn.ReLU())
        self.feat2 = nn.Sequential(nn.Linear(32*5*5, 32*12), nn.ReLU())
        self.linear = nn.Sequential(nn.Linear(32*12, 13))

    def forward(self, x):
        x = self.feat1(x)
        x = self.feat2(x)
        return self.linear(x)
