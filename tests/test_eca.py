from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.attention.eca import ECALayer

x = torch.randn(2, 64, 80, 80)

eca = ECALayer(64)

y = eca(x)

print("Input :", x.shape)
print("Output:", y.shape)

assert x.shape == y.shape

print("ECA test passed.")