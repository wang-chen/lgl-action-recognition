import torch.nn as nn


class CNN(nn.Module):    
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(25, 32, kernel_size=20, stride=2, padding=0),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=10, stride=2, padding=0),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4, 13))

    def forward(self, x):
        feature = self.features(x)
        return self.classifier(feature)