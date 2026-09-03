import torch
import torch.nn as nn

from models.common import Conv
from models.lightweight.lwconv import LWConv

class LWBottleneck(nn.Module):

    def __init__(self, c, shortcut=True):
        super().__init__()

        self.cv1 = LWConv(c)
        self.cv2 = LWConv(c)
        self.add = shortcut

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class LWC3(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()

        c_ = int(c2 * e)

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)

        self.m = nn.Sequential(
            *[LWBottleneck(c_, shortcut) for _ in range(n)]
        )

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat((y1, y2), dim=1))