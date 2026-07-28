import torch.nn as nn

from models.common import Conv
from models.attention.eca import ECALayer


class ConvECA(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()

        self.conv = Conv(c1, c2, k, s, p, g, act)
        self.eca = ECALayer(c2)

    def forward(self, x):
        return self.eca(self.conv(x))