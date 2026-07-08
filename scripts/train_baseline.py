"""
This script reads the training configuration and launches
YOLOv5 training.

Future versions will automatically enable:

- CutMix
- Anchor Refinement
- LWConv
- LW_C3
- Wise-IoU
- P2 Head
- AFP
"""

import os
import sys
import yaml
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CONFIG_FILE = ROOT / "configs" / "train" / "train_idd.yaml"


def load_yaml(path):

    with open(path, "r") as f:
        return yaml.safe_load(f)


def print_header():

    print("=" * 70)
    print(" Enhanced YOLOv5s Baseline Training ")
    print("=" * 70)


def verify_paths(cfg):

    dataset_yaml = ROOT / cfg["dataset"]["yaml"]

    model_cfg = ROOT / cfg["model"]["config"]

    weights = ROOT / cfg["model"]["weights"]

    yolov5 = ROOT / "yolov5"

    assert dataset_yaml.exists(), f"Dataset yaml missing:\n{dataset_yaml}"

    assert model_cfg.exists(), f"Model yaml missing:\n{model_cfg}"

    assert weights.exists(), f"Weights missing:\n{weights}"

    assert yolov5.exists(), "YOLOv5 folder not found."

    print("[OK] All required files found.")


def show_configuration(cfg):

    print("\nProject")

    print(f" Name        : {cfg['project']['name']}")

    print(f" Experiment  : {cfg['project']['experiment']}")

    print("\nTraining")

    print(f" Epochs      : {cfg['training']['epochs']}")

    print(f" Batch       : {cfg['training']['batch_size']}")

    print(f" Image Size  : {cfg['training']['image_size']}")

    print(f" Optimizer   : {cfg['training']['optimizer']}")

    print(f" Cache       : {cfg['training']['cache']}")

    print("\nEnhancements")

    for k, v in cfg["enhancements"].items():

        print(f" {k:20} {v}")


def build_command(cfg):

    cmd = [
        sys.executable,
        "train.py",

        "--img",
        str(cfg["training"]["image_size"]),

        "--batch",
        str(cfg["training"]["batch_size"]),

        "--epochs",
        str(cfg["training"]["epochs"]),

        "--workers",
        str(cfg["training"]["workers"]),

        "--device",
        str(cfg["training"]["device"]),

        "--data",
        str(ROOT / cfg["dataset"]["yaml"]),

        "--cfg",
        str(ROOT / cfg["model"]["config"]),

        "--weights",
        str(ROOT / cfg["model"]["weights"]),

        "--project",
        str(ROOT / cfg["project"]["output_dir"]),

        "--name",
        cfg["project"]["experiment"],

        "--exist-ok",
    ]

    if cfg["training"]["cache"]:
        cmd.append("--cache")

    return cmd


def train(cfg):

    yolov5_dir = ROOT / "yolov5"

    cmd = build_command(cfg)

    print("\nLaunching Training...\n")

    subprocess.run(

        cmd,

        cwd=yolov5_dir,

        check=True

    )


def main():

    print_header()

    cfg = load_yaml(CONFIG_FILE)

    verify_paths(cfg)

    show_configuration(cfg)

    train(cfg)

    print("\nTraining Completed.")


if __name__ == "__main__":

    main()