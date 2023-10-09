import torch
from torch import nn


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, layer_shapes=[]):
        super(Generator, self).__init__()
        layers = []

        for i, (s1, s2) in enumerate(layer_shapes):
            layers.append(nn.Linear(s1, s2))
            if i == len(layer_shapes) - 1:
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        x = torch.cat([x, c], dim=1)
        return self.main.float()(x.float())
