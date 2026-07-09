from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yolov5.models.lightweight.lwc3 import LWC3

x = torch.randn(1, 64, 80, 80)

model = LWC3(64, n=3)

y = model(x)

print("Input :", x.shape)
print("Output:", y.shape)

assert x.shape == y.shape

print("LWC3 test passed.")

params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {params:,}")