"""
sync_yolov5_overlay.py

Manages custom YOLOv5 architecture/config overlays cleanly outside the official
`yolov5/` git clone by syncing them to/from `model_changes/` (which is tracked
in the main `multi-road-object-detection` repository).

Usage:
  1. Export local custom changes from `yolov5/` -> `model_changes/` (before git commit):
     python scripts/sync_yolov5_overlay.py --export

  2. Apply overlays on Kaggle / Colab / fresh clone (clones official yolov5 if
     missing and applies all custom modules, hyps, configs, and weights):
     python scripts/sync_yolov5_overlay.py --apply
"""

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]
YOLO_DIR = ROOT / "yolov5"
OVERLAY_DIR = ROOT / "model_changes"
OFFICIAL_YOLOV5_REPO = "https://github.com/ultralytics/yolov5.git"

# Relative paths inside `yolov5/` that represent project-specific customizations
OVERLAY_FILES: List[str] = [
    "models/common.py",
    "models/experimental.py",
    "models/tf.py",
    "models/yolo.py",
    "models/enhanced_rdd4d.yaml",
    "models/traffic_sign_enhanced.yaml",
    "models/yolov5s_p2.yaml",
    "models/yolov5s_p2_lwc3.yaml",
    "models/yolov5s_p2_lwc3_eca.yaml",
    "models/yolov5s_p2_lwc3_eca_afp.yaml",
    "utils/loss.py",
    "data/hyps/hyp.traffic_sign_finetune.yaml",
    "data/hyps/hyp.road_damage_finetune.yaml",
    "data/road_damage.yaml",
    "data/traffic_sign.yaml",
]

OVERLAY_DIRS: List[str] = [
    "models/attention",
    "models/fusion",
    "models/lightweight",
    "models/losses",
]

BASE_WEIGHTS: List[str] = [
    "runs/train/traffic_sign_enhanced_1280_pretrained/weights/best.pt",
    "runs/train/enhanced_rdd2022_final/weights/best.pt",
    "runs/train/rdd4d_enhanced_final/weights/best.pt",
    "runs/train/idd_final/weights/best.pt",
]


def _copy_file(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _copy_py_tree(src_dir: Path, dst_dir: Path) -> int:
    if not src_dir.exists():
        return 0
    copied = 0
    for src in src_dir.rglob("*"):
        if not src.is_file():
            continue
        if "__pycache__" in src.parts or src.suffix == ".pyc":
            continue
        rel = src.relative_to(src_dir)
        dst = dst_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return copied


def export_overlay(yolo_dir: Path = YOLO_DIR, overlay_dir: Path = OVERLAY_DIR) -> None:
    print(f"Exporting custom YOLOv5 files: {yolo_dir} -> {overlay_dir}")
    file_count = 0

    for rel in OVERLAY_FILES:
        if _copy_file(yolo_dir / rel, overlay_dir / rel):
            file_count += 1
            print(f"  [file] {rel}")
        else:
            print(f"  [skip - missing] {rel}")

    for rel_dir in OVERLAY_DIRS:
        n = _copy_py_tree(yolo_dir / rel_dir, overlay_dir / rel_dir)
        file_count += n
        print(f"  [dir ] {rel_dir} ({n} files)")

    print(f"\nExport complete: {file_count} custom files synced into {overlay_dir}.")


def apply_overlay(yolo_dir: Path = YOLO_DIR, overlay_dir: Path = OVERLAY_DIR) -> None:
    # Step 1: Clone official ultralytics/yolov5 if yolov5/ is missing or empty
    if not (yolo_dir / "train.py").exists():
        if yolo_dir.exists() and not any(yolo_dir.iterdir()):
            yolo_dir.rmdir()
        print(f"Cloning official YOLOv5 into {yolo_dir}...")
        subprocess.run(
            ["git", "clone", "--depth", "1", OFFICIAL_YOLOV5_REPO, str(yolo_dir)],
            check=True,
        )

    print(f"Applying custom overlays: {overlay_dir} -> {yolo_dir}")
    file_count = 0

    for rel in OVERLAY_FILES:
        if _copy_file(overlay_dir / rel, yolo_dir / rel):
            file_count += 1
            print(f"  [file] {rel}")

    for rel_dir in OVERLAY_DIRS:
        n = _copy_py_tree(overlay_dir / rel_dir, yolo_dir / rel_dir)
        file_count += n
        print(f"  [dir ] {rel_dir} ({n} files)")

    # Copy base pretrained weights if present in model_changes/runs/train/
    weights_copied = 0
    for rel_w in BASE_WEIGHTS:
        dst_w = yolo_dir / rel_w
        src_w = overlay_dir / rel_w
        if not dst_w.exists() and src_w.exists():
            _copy_file(src_w, dst_w)
            weights_copied += 1
            print(f"  [ckpt] {rel_w}")

    print(
        f"\nOverlay applied: {file_count} source/config files + "
        f"{weights_copied} checkpoint weights synced into {yolo_dir}."
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Sync custom YOLOv5 overlays between yolov5/ and model_changes/")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--export", action="store_true", help="Copy custom files from yolov5/ to model_changes/")
    group.add_argument("--apply", action="store_true", help="Clone official yolov5 if needed and apply model_changes/")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.export:
        export_overlay()
    elif args.apply:
        apply_overlay()


if __name__ == "__main__":
    main()
