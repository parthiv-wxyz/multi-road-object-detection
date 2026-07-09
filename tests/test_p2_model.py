from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yolov5.models.yolo import Model

model = Model("yolov5/models/yolov5s_p2.yaml")

model.eval()

x = torch.randn(1, 3, 640, 640)

with torch.no_grad():
    y = model(x)

print("=" * 60)
print("P2 MODEL TEST")
print("=" * 60)

print("Model loaded successfully.")
print("Input shape :", x.shape)

if isinstance(y, (list, tuple)):
    print(f"Returned {len(y)} output(s)")
    for i, out in enumerate(y):
        if torch.is_tensor(out):
            print(f"Output {i}: {out.shape}")

print("\nForward pass successful.")