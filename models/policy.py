import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.nn.utils as utils
import math


def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


class Policy_CNN_discrete(nn.Module):
    def __init__(self, switches=4, in_channels=64):
        super(Policy_CNN_discrete, self).__init__()
        self.switches = switches
        hidden_dim = 10

        self.reduce = nn.Sequential(
            conv_dw(in_channels, hidden_dim, 1),
            conv_dw(hidden_dim, hidden_dim, 1),
            conv_dw(hidden_dim, hidden_dim, 1),
            conv_dw(hidden_dim, hidden_dim, 1),
            conv_dw(hidden_dim, hidden_dim, 1),
            conv_dw(hidden_dim, hidden_dim, 1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier_a = nn.Linear(hidden_dim, switches)
        self.classifier_w = nn.Linear(hidden_dim, switches)

    def forward(self, x):
        x = self.reduce(x)
        x = torch.flatten(x, 1)
        logits_a = self.classifier_a(x)
        logits_w = self.classifier_w(x)
        probs_a = F.softmax(logits_a, dim=1)
        probs_w = F.softmax(logits_w, dim=1)

        return probs_a, probs_w


class Policy_CNN_continuous(nn.Module):
    def __init__(self, switches=4, in_channels=64):
        super(Policy_CNN_continuous, self).__init__()
        self.switches = switches
        hidden_dim = 10

        self.reduce = nn.Sequential(
            conv_dw(in_channels, hidden_dim, 1),
            conv_dw(hidden_dim, hidden_dim, 1),
            conv_dw(hidden_dim, hidden_dim, 1),
            conv_dw(hidden_dim, hidden_dim, 1),
            conv_dw(hidden_dim, hidden_dim, 1),
            conv_dw(hidden_dim, hidden_dim, 1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier_a = nn.Linear(hidden_dim, 1)
        self.classifier_w = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.reduce(x)
        x = torch.flatten(x, 1)
        bit_a = self.classifier_a(x)
        bit_w = self.classifier_w(x)
        bit_a = discretize(bit_a)
        bit_w = discretize(bit_w)

        return bit_a, bit_w


class discretize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x[x <= 0.5] = 0
        x[(x > 0.5) * (x <= 1.5)] = 1
        x[(x > 1.5) * (x <= 2.5)] = 2
        x[(x > 2.5) * (x <= 5)] = 3
        x[x > 5] = 8
        return x

    @staticmethod
    def backward(ctx, g):
        return g
