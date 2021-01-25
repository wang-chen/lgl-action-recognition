import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat1 = nn.Conv1d(25, 32, kernel_size=11, stride=1, padding=5)
        self.acvt1 = nn.Sequential(nn.BatchNorm1d(32), nn.MaxPool1d(2), nn.ReLU())
        self.feat2 = nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2)
        self.acvt2 = nn.Sequential(nn.BatchNorm1d(32), nn.MaxPool1d(2), nn.ReLU())
        self.linear = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(32 * 12, 13))

    def forward(self, x):
        x = x.view(x.size(0), -1, x.size(-1))
        x = self.feat1(x)
        x = self.acvt1(x)
        x = self.feat2(x)
        x = self.acvt2(x)
        return self.linear(x)
