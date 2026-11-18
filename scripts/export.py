"""
export.py

Exports trained YOLO PyTorch models (.pt) to optimized formats (ONNX, TorchScript).

Usage:
    python scripts/export.py --weights yolov5/runs/train/idd_final/weights/best.pt --include onnx torchscript --img 640
"""

import argparse
from pathlib import Path
import subprocess
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX or TorchScript")
    parser.add_argument("--weights", type=str, default="yolov5/runs/train/idd_final/weights/best.pt", help="Path to .pt weights")
    parser.add_argument("--img", type=int, default=640, help="Image size")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--include", nargs="+", default=["torchscript", "onnx"], help="Formats: torchscript onnx engine")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    return parser.parse_args()

def main():
    args = parse_args()
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    yolo_export = Path("yolov5") / "export.py"
    if not yolo_export.exists():
        raise FileNotFoundError(f"YOLOv5 export script not found: {yolo_export}")

    cmd = [
        sys.executable,
        str(yolo_export),
        "--weights", str(weights_path),
        "--img", str(args.img),
        "--batch-size", str(args.batch_size),
        "--opset", str(args.opset),
        "--include", *args.include,
    ]

    print("=" * 65)
    print("EXPORTING YOLO MODEL")
    print("=" * 65)
    print(f"Weights: {weights_path}")
    print(f"Targets: {args.include}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 65)

    res = subprocess.run(cmd)
    if res.returncode == 0:
        print("\nExport completed successfully!")
    else:
        print(f"\nExport exited with code {res.returncode}")
        sys.exit(res.returncode)

if __name__ == "__main__":
    main()
