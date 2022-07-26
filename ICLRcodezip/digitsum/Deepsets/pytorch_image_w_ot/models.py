from torch import nn
from deepsetlayers import InvLinear
import torch
import numpy as np


def default_aggregator(x):
    out = torch.sum(x, dim=1, keepdim=False)
    return out


class MNIST_AdderCNN(nn.Module):
    def __init__(self):
        super(MNIST_AdderCNN, self).__init__()

        center_dps = torch.from_numpy(np.random.randn(10, 128)).cuda()
        self.center_dps = nn.Parameter(center_dps.clone().detach().requires_grad_(True))

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.ReLU(inplace=True),
        )
        self.mlp1 = nn.Sequential(
            nn.Linear(2*2*64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(128, 8),
            nn.ReLU(inplace=True)
        )
        self.ot_mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10),
            nn.Softmax(dim=1)
        )
        self.adder = InvLinear(8, 1, reduction='sum', bias=True)

    def forward(self, X, mask=None):
        N, S, C, D, _ = X.shape
        h = self.feature_extractor(X.reshape(N*S, C, D, D))
        h = self.mlp1(h.reshape(N, S, -1))
        y = self.adder(self.mlp2(h), mask=mask)
        ot = self.ot_mlp(default_aggregator(h))
        return y, h, ot
