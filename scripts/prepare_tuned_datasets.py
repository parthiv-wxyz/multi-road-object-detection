"""
prepare_tuned_datasets.py

Builds domain-tuned YOLO datasets with cross-domain hard negatives and optional
multi-dataset merging for fine-tuning:
  1. Traffic Sign detector (`traffic_sign_tuned`)
     - Base: Indian Roads Traffic Signs dataset (5 classes)
     - Hard Negatives: 20% IDD vehicle/road scenes with empty `.txt` labels
       (teaches the Traffic Sign objectness head to stop firing on vehicles!)
  2. Road Damage detector (`road_damage_tuned`)
     - Base: RDD 2022 dataset (YOLO `.txt` or Pascal VOC `.xml`, 5 classes)
     - Optional Extra Dataset: Merges additional pothole YOLO datasets (class -> 4)
     - Hard Negatives: 10% IDD vehicle/clean-road scenes with empty `.txt` labels
       (prevents false cracks on vehicle bodies, shadows, and sky!)

Works out-of-the-box on Kaggle (`/kaggle/input/`), Google Colab, and local machines.
"""

import argparse
import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

RDD_XML_CLASS_MAP: Dict[str, int] = {
    "d00": 0,  # longitudinal crack
    "d01": 0,
    "d10": 1,  # transverse crack
    "d11": 1,
    "d20": 2,  # alligator crack
    "d40": 4,  # pothole
    "d43": 3,  # other corruption / crosswalk blur
    "d44": 3,  # other corruption / white line blur
    "d50": 3,  # utility cover / other corruption
    "repair": 3,
}

EXTRA_ROAD_XML_CLASS_MAP: Dict[str, int] = {
    **RDD_XML_CLASS_MAP,
    "pothole": 4,
    "potholes": 4,
    "pot hole": 4,
    "pot_hole": 4,
    "longitudinal": 0,
    "longitudinal crack": 0,
    "crack": 0,
    "transverse": 1,
    "transversal": 1,
    "transverse crack": 1,
    "alligator": 2,
    "alligator crack": 2,
    "block": 2,
    "block crack": 2,
    "patch": 3,
    "other corruption": 3,
}


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def _find_all_images(root: Path) -> List[Path]:
    if not root or not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def _locate_label_for_image(img_path: Path) -> Tuple[Optional[Path], str]:
    """
    Find matching YOLO `.txt` or Pascal VOC `.xml` annotation for `img_path`.
    """
    # 1. Sibling file
    txt_sibling = img_path.with_suffix(".txt")
    if txt_sibling.exists():
        return txt_sibling, "txt"
    xml_sibling = img_path.with_suffix(".xml")
    if xml_sibling.exists():
        return xml_sibling, "xml"

    # 2. Replace 'images' folder segment with 'labels' or 'annotations'
    parts = list(img_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            for rep, ext, kind in [
                ("labels", ".txt", "txt"),
                ("annotations", ".xml", "xml"),
                ("annotations/xmls", ".xml", "xml"),
            ]:
                candidate_parts = parts[:i] + rep.split("/") + parts[i + 1 :]
                cand = Path(*candidate_parts).with_suffix(ext)
                if cand.exists():
                    return cand, kind

    return None, ""


def _convert_voc_xml_to_yolo(xml_path: Path, class_map: Dict[str, int]) -> List[str]:
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return []

    size = root.find("size")
    if size is None:
        return []
    w = float(size.findtext("width", default="0"))
    h = float(size.findtext("height", default="0"))
    if w <= 0 or h <= 0:
        return []

    lines: List[str] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name not in class_map:
            continue
        cls_id = class_map[name]
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        xmin = max(0.0, min(w, float(bnd.findtext("xmin", default="0"))))
        ymin = max(0.0, min(h, float(bnd.findtext("ymin", default="0"))))
        xmax = max(0.0, min(w, float(bnd.findtext("xmax", default="0"))))
        ymax = max(0.0, min(h, float(bnd.findtext("ymax", default="0"))))
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        if bw <= 0.002 or bh <= 0.002:
            continue
        xc = (xmin + xmax) / (2.0 * w)
        yc = (ymin + ymax) / (2.0 * h)
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines


def _sanitize_yolo_lines(
    txt_path: Path,
    valid_nc: int,
    remap_all_to_class: Optional[int] = None,
) -> List[str]:
    clean: List[str] = []
    try:
        raw_lines = txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []

    for line in raw_lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:5])
        except ValueError:
            continue
        if remap_all_to_class is not None:
            cls_id = remap_all_to_class
        if not (0 <= cls_id < valid_nc):
            continue
        if not (0.0 < bw <= 1.0 and 0.0 < bh <= 1.0 and 0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
            continue
        clean.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return clean


def _infer_split(img_path: Path) -> Optional[str]:
    lower_parts = [p.lower() for p in img_path.parts]
    if any(k in lower_parts for k in ("val", "valid", "validation")):
        return "val"
    if "test" in lower_parts:
        return "val"
    if "train" in lower_parts:
        return "train"
    return None


def _add_hard_negatives(
    idd_images: List[Path],
    out_dir: Path,
    count: int,
    prefix: str,
    rng: random.Random,
) -> int:
    if not idd_images or count <= 0:
        return 0

    chosen = rng.sample(idd_images, min(count, len(idd_images)))
    added = 0
    for idx, img in enumerate(chosen):
        stem = f"{prefix}_neg_{idx:05d}"
        dst_img = out_dir / "images" / "train" / f"{stem}{img.suffix.lower()}"
        dst_lbl = out_dir / "labels" / "train" / f"{stem}.txt"
        _link_or_copy(img, dst_img)
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.write_text("", encoding="utf-8")  # empty label = hard negative background
        added += 1
    return added


def _ensure_val_split(out_dir: Path, counts: Dict[str, int], rng: random.Random, val_ratio: float = 0.15) -> None:
    """If the source dataset had no validation split, carve out val_ratio from train."""
    if counts["val"] > 0 or counts["train"] < 2:
        return
    train_imgs = sorted((out_dir / "images" / "train").glob("*"))
    n_val = max(1, int(len(train_imgs) * val_ratio))
    chosen = rng.sample(train_imgs, n_val)
    for img in chosen:
        dst_img = out_dir / "images" / "val" / img.name
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(img), str(dst_img))

        src_lbl = out_dir / "labels" / "train" / f"{img.stem}.txt"
        if src_lbl.exists():
            dst_lbl = out_dir / "labels" / "val" / src_lbl.name
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_lbl), str(dst_lbl))

        counts["train"] -= 1
        counts["val"] += 1


def prepare_traffic_sign_dataset(
    sign_dir: Path,
    idd_images: List[Path],
    out_dir: Path,
    hard_neg_ratio: float = 0.22,
    seed: int = 42,
) -> Path:
    rng = random.Random(seed)
    images = _find_all_images(sign_dir)
    if not images:
        raise FileNotFoundError(f"No traffic sign images found in {sign_dir}")

    counts = {"train": 0, "val": 0}
    for idx, img in enumerate(images):
        lbl_path, kind = _locate_label_for_image(img)
        if lbl_path is None or kind != "txt":
            continue

        lines = _sanitize_yolo_lines(lbl_path, valid_nc=5)
        split = _infer_split(img)
        if split is None:
            split = "val" if rng.random() < 0.18 else "train"

        stem = f"ts_{idx:06d}_{img.stem}"
        dst_img = out_dir / "images" / split / f"{stem}{img.suffix.lower()}"
        dst_lbl = out_dir / "labels" / split / f"{stem}.txt"
        _link_or_copy(img, dst_img)
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.write_text("\n".join(lines), encoding="utf-8")
        counts[split] += 1

    _ensure_val_split(out_dir, counts, rng, val_ratio=0.15)

    # Add IDD vehicle hard negatives into train split so Traffic Sign stops firing on vehicles
    neg_target = int(counts["train"] * hard_neg_ratio)
    neg_added = _add_hard_negatives(idd_images, out_dir, neg_target, "idd_veh", rng)

    yaml_path = out_dir / "traffic_sign_tuned.yaml"
    cfg = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 5,
        "names": {
            0: "regulatory",
            1: "mandatory",
            2: "informatory",
            3: "general",
            4: "warning",
        },
    }
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(
        f"[Traffic Sign Tuned] train={counts['train']} + {neg_added} IDD vehicle hard negatives | "
        f"val={counts['val']} -> {yaml_path}"
    )
    return yaml_path


def prepare_road_damage_dataset(
    rdd_dir: Path,
    idd_images: List[Path],
    out_dir: Path,
    extra_pothole_dir: Optional[Path] = None,
    hard_neg_ratio: float = 0.10,
    max_rdd_images: Optional[int] = None,
    seed: int = 42,
) -> Path:
    rng = random.Random(seed)
    images = _find_all_images(rdd_dir)
    if not images:
        raise FileNotFoundError(f"No RDD images found in {rdd_dir}")

    if max_rdd_images and len(images) > max_rdd_images:
        rng.shuffle(images)
        images = images[:max_rdd_images]

    counts = {"train": 0, "val": 0}
    for idx, img in enumerate(images):
        lbl_path, kind = _locate_label_for_image(img)
        if lbl_path is None:
            continue

        if kind == "xml":
            lines = _convert_voc_xml_to_yolo(lbl_path, RDD_XML_CLASS_MAP)
        else:
            lines = _sanitize_yolo_lines(lbl_path, valid_nc=5)

        split = _infer_split(img)
        if split is None:
            split = "val" if rng.random() < 0.15 else "train"

        stem = f"rdd_{idx:06d}_{img.stem}"
        dst_img = out_dir / "images" / split / f"{stem}{img.suffix.lower()}"
        dst_lbl = out_dir / "labels" / split / f"{stem}.txt"
        _link_or_copy(img, dst_img)
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.write_text("\n".join(lines), encoding="utf-8")
        counts[split] += 1

    # Optionally merge an extra pothole/crack dataset (supports both Pascal VOC XML
    # like `andrewmvd/pothole-detection` and YOLO `.txt` pothole datasets)
    extra_added = 0
    if extra_pothole_dir and extra_pothole_dir.exists():
        pothole_imgs = _find_all_images(extra_pothole_dir)
        for idx, img in enumerate(pothole_imgs):
            lbl_path, kind = _locate_label_for_image(img)
            if lbl_path is None:
                continue
            if kind == "xml":
                lines = _convert_voc_xml_to_yolo(lbl_path, EXTRA_ROAD_XML_CLASS_MAP)
            else:
                lines = _sanitize_yolo_lines(lbl_path, valid_nc=5, remap_all_to_class=4)
            if not lines:
                continue
            split = "val" if rng.random() < 0.12 else "train"
            stem = f"pothole_ext_{idx:05d}_{img.stem}"
            dst_img = out_dir / "images" / split / f"{stem}{img.suffix.lower()}"
            dst_lbl = out_dir / "labels" / split / f"{stem}.txt"
            _link_or_copy(img, dst_img)
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.write_text("\n".join(lines), encoding="utf-8")
            counts[split] += 1
            extra_added += 1

    _ensure_val_split(out_dir, counts, rng, val_ratio=0.15)

    # Add IDD clean/vehicle hard negatives so road damage doesn't fire on cars or sky
    neg_target = int(counts["train"] * hard_neg_ratio)
    neg_added = _add_hard_negatives(idd_images, out_dir, neg_target, "idd_bg", rng)

    yaml_path = out_dir / "road_damage_tuned.yaml"
    cfg = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 5,
        "names": {
            0: "longitudinal crack",
            1: "transverse crack",
            2: "alligator crack",
            3: "other corruption",
            4: "pothole",
        },
    }
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(
        f"[Road Damage Tuned] train={counts['train']} (incl. {extra_added} extra pothole imgs) "
        f"+ {neg_added} IDD hard negatives | val={counts['val']} -> {yaml_path}"
    )
    return yaml_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare tuned Traffic Sign & Road Damage datasets with cross-domain hard negatives"
    )
    parser.add_argument("--idd-dir", type=str, default="", help="Path to IDD dataset (source of hard negatives)")
    parser.add_argument("--traffic-sign-dir", type=str, default="", help="Path to Indian Traffic Signs dataset")
    parser.add_argument("--rdd-dir", type=str, default="", help="Path to RDD 2022 dataset")
    parser.add_argument("--extra-pothole-dir", type=str, default="", help="Optional extra pothole YOLO dataset")
    parser.add_argument("--output-root", type=str, default="./prepared_tuned_datasets", help="Output root directory")
    parser.add_argument("--sign-neg-ratio", type=float, default=0.22, help="IDD hard-negative ratio for Traffic Sign")
    parser.add_argument("--rdd-neg-ratio", type=float, default=0.10, help="IDD hard-negative ratio for Road Damage")
    parser.add_argument("--max-rdd-images", type=int, default=0, help="Optional cap on RDD images (0 = all)")
    return parser.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.output_root)
    idd_images = _find_all_images(Path(args.idd_dir)) if args.idd_dir else []
    print(f"Found {len(idd_images)} IDD images available for hard-negative sampling.")

    if args.traffic_sign_dir:
        prepare_traffic_sign_dataset(
            sign_dir=Path(args.traffic_sign_dir),
            idd_images=idd_images,
            out_dir=out_root / "traffic_sign_tuned",
            hard_neg_ratio=args.sign_neg_ratio,
        )

    if args.rdd_dir:
        prepare_road_damage_dataset(
            rdd_dir=Path(args.rdd_dir),
            idd_images=idd_images,
            out_dir=out_root / "road_damage_tuned",
            extra_pothole_dir=Path(args.extra_pothole_dir) if args.extra_pothole_dir else None,
            hard_neg_ratio=args.rdd_neg_ratio,
            max_rdd_images=args.max_rdd_images if args.max_rdd_images > 0 else None,
        )


if __name__ == "__main__":
    main()
