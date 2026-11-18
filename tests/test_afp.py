from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "yolov5"))

from models.fusion.afp import AFP, ChannelAttention, SpatialAttention

print("=" * 60)
print("AFP (Attention Feature Pyramid) Unit Tests")
print("=" * 60)

# 1. Test Channel Attention
ca = ChannelAttention(channels=64, reduction=16)
x = torch.randn(2, 64, 40, 40)
y_ca = ca(x)
assert y_ca.shape == x.shape, f"ChannelAttention shape mismatch: {y_ca.shape} vs {x.shape}"
print("ChannelAttention test passed.")

# 2. Test Spatial Attention
sa = SpatialAttention()
y_sa = sa(x)
assert y_sa.shape == x.shape, f"SpatialAttention shape mismatch: {y_sa.shape} vs {x.shape}"
print("SpatialAttention test passed.")

# 3. Test AFP Module (fusion of two feature maps)
afp = AFP(c1=64, c2=64)
f1 = torch.randn(2, 64, 40, 40)
f2 = torch.randn(2, 64, 40, 40)
fused = afp([f1, f2])

print(f"Inputs : f1={f1.shape}, f2={f2.shape}")
print(f"Output : {fused.shape}")
assert fused.shape == (2, 64, 40, 40), f"AFP output shape mismatch: {fused.shape}"
print("AFP fusion test passed.")
