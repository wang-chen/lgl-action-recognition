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
