import torch
import torch.nn as nn


class LWConv(nn.Module):
    """
    Lightweight Multi-Kernel Convolution

    Splits channels into four groups and applies:
    1x1, 3x3, 5x5 and 7x7 convolutions,
    followed by a 1x1 pointwise fusion.
    """

    def __init__(self, channels):
        super().__init__()

        assert channels % 4 == 0, "Channels must be divisible by 4."

        g = channels // 4

        self.conv1 = nn.Conv2d(g, g, 1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(g, g, 3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(g, g, 5, padding=2, bias=False)
        self.conv7 = nn.Conv2d(g, g, 7, padding=3, bias=False)

        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()

        self.fuse = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):

        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)

        y1 = self.conv1(x1)
        y2 = self.conv3(x2)
        y3 = self.conv5(x3)
        y4 = self.conv7(x4)

        y = torch.cat([y1, y2, y3, y4], dim=1)

        y = self.bn(y)
        y = self.act(y)

        return self.fuse(y)