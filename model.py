import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, (5, 5), bias=True, padding=int(5/2))
        self.conv2 = nn.Conv2d(32, 32, (5, 5), bias=True, padding=int(5/2))
        self.conv3 = nn.Conv2d(32, 32, (5, 5), bias=True, padding=int(5/2))
        self.out = nn.Conv2d(32, 1, (5, 5), bias=True, padding=int(5/2))

    def forward(self, x):
        x = torch.relu(self.conv1(x))

        x = torch.relu(self.conv2(x))

        x = torch.relu(self.conv3(x))
        return self.out(x)
