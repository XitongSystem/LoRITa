import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange
from collections import OrderedDict

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, factor = 1):
        super().__init__()
        self.norm = nn.BatchNorm1d(hidden_dim)
        if factor == 1:
            self.net1 = nn.Linear(dim, hidden_dim)

        else:
            self.net1 = OrderedDict()
            for i in range(factor - 1):
                self.net1['compress_'+str(i)] = nn.Linear(dim, dim, bias=False)
            self.net1['compress_'+str(factor - 1)] = nn.Linear(dim, hidden_dim)
            self.net1 = nn.Sequential(self.net1)

    def forward(self, x):
        x = self.net1(x)
        x = self.norm(x)
        return x

class FCN(nn.Module):
    def __init__(self, *, image_size, num_classes, dim, depth, factor=1):
        super().__init__()
        
        self.layers = [FeedForward(image_size, dim, factor), nn.ReLU()]
        for i in range(depth - 2):
            self.layers.append(FeedForward(dim, dim, factor))
            self.layers.append(nn.ReLU())
        self.layers.append(FeedForward(dim, num_classes, factor))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, img):
        return self.layers(img.view(img.shape[0],-1))