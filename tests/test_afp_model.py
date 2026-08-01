from pathlib import Path
import sys
import torch

# ------------------------------------------------------------------
# Add YOLOv5 directory to Python path
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "yolov5"))

# ------------------------------------------------------------------
# Import YOLOv5 Model
# ------------------------------------------------------------------
from models.yolo import Model

# ------------------------------------------------------------------
# Load model configuration
# ------------------------------------------------------------------
CFG = ROOT / "yolov5" / "models" / "yolov5s_p2_lwc3_eca_afp.yaml"

model = Model(str(CFG))
model.eval()

print("=" * 60)
print("LW_C3 + ECA MODEL TEST")
print("=" * 60)

print("Model loaded successfully.")

# ------------------------------------------------------------------
# Dummy input
# ------------------------------------------------------------------
x = torch.randn(1, 3, 640, 640)

print(f"Input shape : {x.shape}")

# ------------------------------------------------------------------
# Forward pass
# ------------------------------------------------------------------
with torch.no_grad():
    outputs = model(x)

# ------------------------------------------------------------------
# Display outputs
# ------------------------------------------------------------------
if isinstance(outputs, (tuple, list)):
    print(f"Returned {len(outputs)} output(s)")

    for i, out in enumerate(outputs):
        if torch.is_tensor(out):
            print(f"Output {i}: {out.shape}")
        elif isinstance(out, (tuple, list)):
            print(f"Output {i}:")
            for j, t in enumerate(out):
                if torch.is_tensor(t):
                    print(f"    [{j}] {t.shape}")
else:
    print(outputs.shape)

print("\nForward pass successful.")