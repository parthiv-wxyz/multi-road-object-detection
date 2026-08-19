"""
Train the enhanced YOLOv5s model
(P2 + LWC3 + ECA + Wise-IoU)

Run:

python scripts/train_enhanced.py
"""

from pathlib import Path
import subprocess
import sys
import torch


ROOT = Path(__file__).resolve().parents[1]

YOLO = ROOT / "yolov5"

CFG = YOLO / "models" / "yolov5s_p2_lwc3_eca.yaml"

DATA = ROOT / "datasets" / "idd" / "data.yaml"

PROJECT = ROOT / "runs"

NAME = "enhanced"

WEIGHTS = YOLO / "yolov5s.pt"

IMG = 640

BATCH = 16

EPOCHS = 100

DEVICE = "0"


# ============================================================
# GPU CHECK
# ============================================================

print("=" * 60)
print("GPU CHECK")
print("=" * 60)

print(f"Python: {sys.executable}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. Install CUDA-enabled PyTorch before training."
    )

print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# TRAINING COMMAND
# ============================================================

cmd = [
    sys.executable,
    str(YOLO / "train.py"),

    "--img", str(IMG),

    "--batch", str(BATCH),

    "--epochs", str(EPOCHS),

    "--data", str(DATA),

    "--cfg", str(CFG),

    "--weights", str(WEIGHTS),

    "--project", str(PROJECT),

    "--name", NAME,

    "--device", DEVICE,

    "--workers", "0",

]


# ============================================================
# START TRAINING
# ============================================================

print("=" * 60)
print("TRAINING ENHANCED YOLOv5")
print("=" * 60)

print(" ".join(map(str, cmd)))

subprocess.run(cmd, check=True)