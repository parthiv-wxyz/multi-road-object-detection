import torch.nn as nn

from models.common import Conv
from models.attention.eca import ECALayer


class ConvECA(nn.Module):
    """
    Conv + ECA block.
    Drop-in replacement for YOLOv5 Conv.
    """

    def __init__(
        self,
        c1,
        c2,
        k=1,
        s=1,
        p=None,
        g=1,
        d=1,
        act=True,
    ):
        super().__init__()

        self.conv = Conv(
            c1=c1,
            c2=c2,
            k=k,
            s=s,
            p=p,
            g=g,
            d=d,
            act=act,
        )

        self.eca = ECALayer(c2)

    def forward(self, x):
        return self.eca(self.conv(x))