"""
Inference Script

Runs inference on images using the trained model.
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
    return ROOT / "runs" / "train" / exp / "weights" / "best.pt"


def detect(cfg):

    weights = get_best_weights(cfg)

    source = ROOT / "datasets" / "idd" / "test" / "images"

    if not source.exists():
        raise FileNotFoundError(source)

    command = [
        sys.executable,
        "detect.py",
        "--weights",
        str(weights),
        "--source",
        str(source),
        "--img",
        str(cfg["training"]["image_size"]),
        "--conf",
        str(cfg["validation"]["conf_threshold"]),
        "--name",
        cfg["project"]["experiment"] + "_detect",
        "--project",
        str(ROOT / "runs" / "detect"),

        "--name",
        cfg["project"]["experiment"],

        "--exist-ok",
    ]

    print("=" * 70)
    print("RUNNING INFERENCE")
    print("=" * 70)

    subprocess.run(
        command,
        cwd=ROOT / "yolov5",
        check=True
    )


def main():
    cfg = load_yaml(CONFIG_FILE)
    detect(cfg)


if __name__ == "__main__":
    main()