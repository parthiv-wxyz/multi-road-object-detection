"""
Train the enhanced YOLOv5s model
(P2 + LWC3 + ECA + Wise-IoU)

Run:

python scripts/train_enhanced.py
"""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

YOLO = ROOT / "yolov5"

CFG = YOLO / "models" / "yolov5s_p2_lwc3_eca.yaml"

DATA = ROOT / "datasets" / "idd" / "data.yaml"

PROJECT = ROOT / "runs"

NAME = "enhanced"

WEIGHTS = "yolov5s.pt"

IMG = 640

BATCH = 16

EPOCHS = 100

DEVICE = "0"

cmd = [
    sys.executable,
    str(YOLO / "train.py"),

    "--img", str(IMG),

    "--batch", str(BATCH),

    "--epochs", str(EPOCHS),

    "--data", str(DATA),

    "--cfg", str(CFG),

    "--weights", WEIGHTS,

    "--project", str(PROJECT),

    "--name", NAME,

    "--device", DEVICE,

    "--workers", "8",

    "--cache"
]

print("=" * 60)
print("TRAINING ENHANCED YOLOv5")
print("=" * 60)

print(" ".join(cmd))

subprocess.run(cmd)