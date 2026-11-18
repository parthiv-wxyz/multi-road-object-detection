"""
finetune_models.py

Two-stage fine-tuning runner for:
  1. Traffic Sign detector (`traffic_sign_enhanced_1280_pretrained/weights/best.pt`)
     - Stage 1: Frozen backbone (`--freeze 10`, 15 epochs @ 1280px, `fliplr=0.0`)
     - Stage 2: Full unfreeze (`--freeze 0`, 15 epochs @ 1280px, cosine LR)
  2. Road Damage detector (`enhanced_rdd2022_final/weights/best.pt`)
     - Stage 1: Frozen backbone (`--freeze 10`, 15 epochs @ 896px, focal loss + anchor_t=5.0)
     - Stage 2: Full unfreeze (`--freeze 0`, 20 epochs @ 896px, multi-scale + cosine LR)

Usage:
    python scripts/finetune_models.py --target both \
        --sign-data prepared_tuned_datasets/traffic_sign_tuned/traffic_sign_tuned.yaml \
        --rdd-data prepared_tuned_datasets/road_damage_tuned/road_damage_tuned.yaml \
        --device 0
"""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
YOLO_DIR = ROOT / "yolov5"
OVERLAY_DIR = ROOT / "model_changes"


def _resolve_existing(primary: Path, fallback: Path) -> Path:
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    return primary


def _ensure_yolov5_ready() -> None:
    if not (YOLO_DIR / "train.py").exists() or not (YOLO_DIR / "models" / "fusion" / "afp.py").exists():
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "sync_yolov5_overlay.py"), "--apply"],
            check=True,
        )


def _run_yolo_train(
    weights: Path,
    data_yaml: Path,
    hyp_yaml: Path,
    imgsz: int,
    batch: int,
    epochs: int,
    freeze: int,
    project: Path,
    name: str,
    device: str,
    multi_scale: bool = False,
) -> Path:
    cmd = [
        sys.executable,
        str(YOLO_DIR / "train.py"),
        "--weights", str(weights),
        "--data", str(data_yaml),
        "--hyp", str(hyp_yaml),
        "--img", str(imgsz),
        "--batch-size", str(batch),
        "--epochs", str(epochs),
        "--freeze", str(freeze),
        "--cos-lr",
        "--project", str(project),
        "--name", name,
        "--exist-ok",
        "--device", device,
        "--workers", "4",
        "--patience", "15",
    ]
    if multi_scale:
        cmd.append("--multi-scale")

    print("\n" + "=" * 72)
    print(f"RUNNING: {name} (freeze={freeze}, imgsz={imgsz}, epochs={epochs})")
    print("CMD:", " ".join(cmd))
    print("=" * 72)

    res = subprocess.run(cmd, cwd=str(YOLO_DIR))
    if res.returncode != 0:
        raise RuntimeError(f"Training stage '{name}' failed with exit code {res.returncode}")

    best_pt = project / name / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Expected output weights not found at {best_pt}")
    return best_pt


def finetune_traffic_sign(
    base_weights: Path,
    data_yaml: Path,
    out_project: Path,
    device: str,
    batch: int = 8,
    stage1_epochs: int = 15,
    stage2_epochs: int = 15,
) -> Path:
    _ensure_yolov5_ready()
    hyp = _resolve_existing(
        OVERLAY_DIR / "data" / "hyps" / "hyp.traffic_sign_finetune.yaml",
        YOLO_DIR / "data" / "hyps" / "hyp.traffic_sign_finetune.yaml",
    )

    # Stage 1: Freeze backbone so neck/heads adapt to IDD vehicle hard negatives cleanly
    stage1_best = _run_yolo_train(
        weights=base_weights,
        data_yaml=data_yaml,
        hyp_yaml=hyp,
        imgsz=1280,
        batch=batch,
        epochs=stage1_epochs,
        freeze=10,
        project=out_project,
        name="traffic_sign_tuned_stage1",
        device=device,
    )

    # Stage 2: Gentle full-network refinement
    stage2_best = _run_yolo_train(
        weights=stage1_best,
        data_yaml=data_yaml,
        hyp_yaml=hyp,
        imgsz=1280,
        batch=batch,
        epochs=stage2_epochs,
        freeze=0,
        project=out_project,
        name="traffic_sign_tuned_final",
        device=device,
    )
    return stage2_best


def finetune_road_damage(
    base_weights: Path,
    data_yaml: Path,
    out_project: Path,
    device: str,
    batch: int = 16,
    stage1_epochs: int = 15,
    stage2_epochs: int = 20,
) -> Path:
    _ensure_yolov5_ready()
    hyp = _resolve_existing(
        OVERLAY_DIR / "data" / "hyps" / "hyp.road_damage_finetune.yaml",
        YOLO_DIR / "data" / "hyps" / "hyp.road_damage_finetune.yaml",
    )

    # Stage 1: Frozen backbone @ 896px for hairline cracks + focal loss + IDD hard negatives
    stage1_best = _run_yolo_train(
        weights=base_weights,
        data_yaml=data_yaml,
        hyp_yaml=hyp,
        imgsz=896,
        batch=batch,
        epochs=stage1_epochs,
        freeze=10,
        project=out_project,
        name="road_damage_tuned_stage1",
        device=device,
        multi_scale=False,
    )

    # Stage 2: Gentle unfreeze with multi-scale training for scale invariance
    stage2_best = _run_yolo_train(
        weights=stage1_best,
        data_yaml=data_yaml,
        hyp_yaml=hyp,
        imgsz=896,
        batch=max(8, batch // 2),
        epochs=stage2_epochs,
        freeze=0,
        project=out_project,
        name="road_damage_tuned_final",
        device=device,
        multi_scale=True,
    )
    return stage2_best


def parse_args():
    parser = argparse.ArgumentParser(description="Two-stage domain fine-tuning for Traffic Sign & Road Damage")
    parser.add_argument("--target", choices=["traffic_sign", "road_damage", "both"], default="both")
    parser.add_argument("--sign-data", type=str, default="")
    parser.add_argument("--rdd-data", type=str, default="")
    parser.add_argument(
        "--sign-weights",
        type=str,
        default=str(
            _resolve_existing(
                YOLO_DIR / "runs/train/traffic_sign_enhanced_1280_pretrained/weights/best.pt",
                OVERLAY_DIR / "runs/train/traffic_sign_enhanced_1280_pretrained/weights/best.pt",
            )
        ),
    )
    parser.add_argument(
        "--rdd-weights",
        type=str,
        default=str(
            _resolve_existing(
                YOLO_DIR / "runs/train/enhanced_rdd2022_final/weights/best.pt",
                OVERLAY_DIR / "runs/train/enhanced_rdd2022_final/weights/best.pt",
            )
        ),
    )
    parser.add_argument("--output-dir", type=str, default=str(YOLO_DIR / "runs/finetune"))
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--sign-batch", type=int, default=8)
    parser.add_argument("--rdd-batch", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    out_project = Path(args.output_dir).resolve()

    if args.target in ("traffic_sign", "both"):
        if not args.sign_data:
            raise ValueError("--sign-data is required when tuning traffic_sign")
        ts_best = finetune_traffic_sign(
            base_weights=Path(args.sign_weights).resolve(),
            data_yaml=Path(args.sign_data).resolve(),
            out_project=out_project,
            device=args.device,
            batch=args.sign_batch,
        )
        print(f"\n[DONE] Tuned Traffic Sign weights saved to: {ts_best}")

    if args.target in ("road_damage", "both"):
        if not args.rdd_data:
            raise ValueError("--rdd-data is required when tuning road_damage")
        rdd_best = finetune_road_damage(
            base_weights=Path(args.rdd_weights).resolve(),
            data_yaml=Path(args.rdd_data).resolve(),
            out_project=out_project,
            device=args.device,
            batch=args.rdd_batch,
        )
        print(f"\n[DONE] Tuned Road Damage weights saved to: {rdd_best}")


if __name__ == "__main__":
    main()
