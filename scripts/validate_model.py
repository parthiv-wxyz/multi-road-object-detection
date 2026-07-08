"""
Validation Script

Evaluates a trained YOLOv5 model using the selected dataset.
"""

import sys
import yaml
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONFIG_FILE = ROOT / "configs" / "train" / "train_idd.yaml"


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_best_weights(cfg):
    exp = cfg["project"]["experiment"]

    weights = (
        ROOT
        / "runs"
        / "train"
        / exp
        / "weights"
        / "best.pt"
    )

    if not weights.exists():
        raise FileNotFoundError(
            f"\nModel not found:\n{weights}\n"
            "Check the experiment name in train_idd.yaml "
            "or verify that the training output was copied correctly."
        )

    return weights


def validate(cfg):

    weights = get_best_weights(cfg)

    if not weights.exists():
        raise FileNotFoundError(
            f"\nModel not found:\n{weights}\n\nRun training first."
        )

    command = [
        sys.executable,
        "val.py",

        "--weights",
        str(weights),

        "--data",
        str(ROOT / cfg["dataset"]["yaml"]),

        "--img",
        str(cfg["training"]["image_size"]),

        "--batch",
        str(cfg["training"]["batch_size"]),

        "--workers",
        str(cfg["training"]["workers"]),

        "--device",
        str(cfg["training"]["device"]),

        "--task",
        "val",

        "--project",
        str(ROOT / "runs" / "val"),

        "--name",
        cfg["project"]["experiment"],

        "--exist-ok",
    ]

    print("=" * 70)
    print("VALIDATING MODEL")
    print("=" * 70)
    print("Weights :", weights)
    print()

    subprocess.run(
        command,
        cwd=ROOT / "yolov5",
        check=True
    )


def main():
    cfg = load_yaml(CONFIG_FILE)
    validate(cfg)


if __name__ == "__main__":
    main()