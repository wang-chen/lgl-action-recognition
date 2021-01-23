import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=(5,10), stride=2, padding=0),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(1,10), stride=2, padding=0),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 6, 13))

    def forward(self, x):
        feature = self.features(x)
        return self.classifier(feature)


class FeatTrans1d(nn.Module):
    '''
    Temporal Feature Transforming Layer for multi-channel 1D features.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x, neighbor):
        B, C, F = x.shape
        adj = self.feature_adjacency(x, neighbor)
        x = self.transform(x, adj)
        neighbor = torch.stack([self.transform(neighbor[:,i,:,:], adj) for i in range(neighbor.size(1))], dim=1)
        return x, neighbor

    def transform(self, x, adj):
        return self.conv((adj @ x.unsqueeze(-1)).squeeze(-1))

    def feature_adjacency(self, x, y):
        fadj = torch.einsum('bcx,bncy->bcxy', x, y)
        fadj += fadj.transpose(-2, -1)
        return self.row_normalize(self.sgnroot(fadj))

    def sgnroot(self, x):
        return x.sign()*(x.abs().sqrt())

    def row_normalize(self, x):
        x = x / (x.abs().sum(-1, keepdim=True) + 1e-7)
        x[torch.isnan(x)] = 0
        return x


if __name__ == "__main__":
    device = "cuda:0"
    x = torch.randn(16, 5, 50).to(device)
    conv = FeatTrans1d(5, 32, 10, 1, 0).to(device)
    neighbor = torch.randn(16, 5, 5, 50).to(device)
    y = conv(x, neighbor)