import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Channel attention for AFP.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()

        hidden = max(channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))


class SpatialAttention(nn.Module):
    """
    Spatial attention for AFP.
    """

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg = torch.mean(x, dim=1, keepdim=True)
        maximum, _ = torch.max(x, dim=1, keepdim=True)

        attention = torch.cat([avg, maximum], dim=1)

        attention = self.sigmoid(self.conv(attention))

        return x * attention


class AFP(nn.Module):
    """
    Attention Feature Pyramid Fusion.

    Accepts two feature maps with the same spatial resolution
    and produces one fused feature map.

    Input:
        [x1, x2]

    Output:
        fused feature map with c2 channels.
    """

    def __init__(self, c1, c2=None):
        super().__init__()

        if c2 is None:
            c2 = c1

        self.c1 = c1
        self.c2 = c2

        # Fuse the two feature maps.
        self.fuse = nn.Sequential(
            nn.Conv2d(
                c1 * 2,
                c2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

        self.channel_attention = ChannelAttention(c2)

        self.spatial_attention = SpatialAttention()

        self.refine = nn.Conv2d(
            c2,
            c2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(c2)

        self.act = nn.SiLU(inplace=True)

    def forward(self, x):

        if not isinstance(x, (list, tuple)):
            raise TypeError(
                "AFP expects two feature maps as a list or tuple."
            )

        if len(x) != 2:
            raise ValueError(
                f"AFP expects exactly 2 feature maps, got {len(x)}."
            )

        x1, x2 = x

        if x1.shape[2:] != x2.shape[2:]:
            x2 = nn.functional.interpolate(
                x2,
                size=x1.shape[2:],
                mode="nearest",
            )

        if x1.shape[1] != self.c1 or x2.shape[1] != self.c1:
            raise ValueError(
                "AFP input channel mismatch: "
                f"expected {self.c1}, "
                f"got {x1.shape[1]} and {x2.shape[1]}."
            )

        # Feature concatenation
        x = torch.cat([x1, x2], dim=1)

        # Channel reduction / fusion
        x = self.fuse(x)

        # Channel attention
        x = self.channel_attention(x)

        # Spatial attention
        x = self.spatial_attention(x)

        # Feature refinement
        x = self.refine(x)
        x = self.bn(x)
        x = self.act(x)

        return x