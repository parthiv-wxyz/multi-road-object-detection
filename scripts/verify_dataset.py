"""
verify_dataset.py

Dataset verification script for YOLO datasets.

Checks:
---------
✓ Missing labels
✓ Missing images
✓ Empty label files
✓ Corrupted images
✓ Invalid YOLO format
✓ Invalid class IDs
✓ Bounding box range
✓ Duplicate annotations
✓ Class distribution

Author: Enhanced YOLOv5s Project
"""

from pathlib import Path
from PIL import Image
import yaml
import os

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = Path("datasets/idd")

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

REPORT_DIR = Path("results/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD DATA.YAML
# ============================================================

yaml_file = DATASET_PATH / "data.yaml"

if not yaml_file.exists():
    raise FileNotFoundError(f"\nCannot find:\n{yaml_file}")

with open(yaml_file, "r") as f:
    cfg = yaml.safe_load(f)

NUM_CLASSES = cfg["nc"]
CLASS_NAMES = cfg["names"]

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)
print(f"Dataset : {DATASET_PATH}")
print(f"Classes : {NUM_CLASSES}")
print()

for i, name in enumerate(CLASS_NAMES):
    print(f"{i:2d} -> {name}")

print("=" * 60)

# ============================================================
# COUNTERS
# ============================================================

total_images = 0
total_labels = 0

class_distribution = {i: 0 for i in range(NUM_CLASSES)}

missing_labels = []
missing_images = []
empty_labels = []
corrupted_images = []
invalid_annotations = []
invalid_class_ids = []
duplicate_boxes = []

# ============================================================
# FUNCTIONS
# ============================================================


def verify_image(path):
    try:
        img = Image.open(path)
        img.verify()
        return True
    except Exception:
        return False


def verify_label(label_path):
    with open(label_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if len(lines) == 0:
        empty_labels.append(label_path)
        return

    duplicates = set()

    for line_number, line in enumerate(lines, start=1):

        values = line.split()

        if len(values) != 5:
            invalid_annotations.append(
                f"{label_path} | Line {line_number} | {line}"
            )
            continue

        try:
            cls = int(values[0])
            x = float(values[1])
            y = float(values[2])
            w = float(values[3])
            h = float(values[4])

        except ValueError:
            invalid_annotations.append(
                f"{label_path} | Line {line_number} | {line}"
            )
            continue

        if cls < 0 or cls >= NUM_CLASSES:
            invalid_class_ids.append(
                f"{label_path} | Class {cls}"
            )
            continue

        if not (0 <= x <= 1):
            invalid_annotations.append(
                f"{label_path} | Invalid x | {line}"
            )

        if not (0 <= y <= 1):
            invalid_annotations.append(
                f"{label_path} | Invalid y | {line}"
            )

        if not (0 < w <= 1):
            invalid_annotations.append(
                f"{label_path} | Invalid width | {line}"
            )

        if not (0 < h <= 1):
            invalid_annotations.append(
                f"{label_path} | Invalid height | {line}"
            )

        class_distribution[cls] += 1

        if line in duplicates:
            duplicate_boxes.append(
                f"{label_path} | {line}"
            )

        duplicates.add(line)


# ============================================================
# VERIFY DATASET
# ============================================================

splits = ["train", "val", "test"]

for split in splits:

    print(f"\nChecking {split} dataset...")

    image_dir = DATASET_PATH / split / "images"
    label_dir = DATASET_PATH / split / "labels"

    if not image_dir.exists():
        print(f"Missing folder: {image_dir}")
        continue

    if not label_dir.exists():
        print(f"Missing folder: {label_dir}")
        continue

    images = []

    for ext in IMAGE_EXTENSIONS:
        images.extend(image_dir.glob(f"*{ext}"))
        images.extend(image_dir.glob(f"*{ext.upper()}"))

    labels = list(label_dir.glob("*.txt"))

    total_images += len(images)
    total_labels += len(labels)

    # -------------------------
    # Images
    # -------------------------

    for image in images:

        label = label_dir / f"{image.stem}.txt"

        if not label.exists():
            missing_labels.append(image)

        if not verify_image(image):
            corrupted_images.append(image)

    # -------------------------
    # Labels
    # -------------------------

    for label in labels:

        image_exists = False

        for ext in IMAGE_EXTENSIONS:

            img = image_dir / f"{label.stem}{ext}"

            if img.exists():
                image_exists = True
                break

            img = image_dir / f"{label.stem}{ext.upper()}"

            if img.exists():
                image_exists = True
                break

        if not image_exists:
            missing_images.append(label)

        verify_label(label)

# ============================================================
# REPORT
# ============================================================

report = []

report.append("=" * 60)
report.append("DATASET VERIFICATION REPORT")
report.append("=" * 60)

report.append(f"Total Images           : {total_images}")
report.append(f"Total Labels           : {total_labels}")

report.append("")
report.append(f"Missing Labels         : {len(missing_labels)}")
report.append(f"Missing Images         : {len(missing_images)}")
report.append(f"Empty Labels           : {len(empty_labels)}")
report.append(f"Corrupted Images       : {len(corrupted_images)}")
report.append(f"Invalid Annotations    : {len(invalid_annotations)}")
report.append(f"Invalid Class IDs      : {len(invalid_class_ids)}")
report.append(f"Duplicate Annotations  : {len(duplicate_boxes)}")

report.append("")
report.append("=" * 60)
report.append("CLASS DISTRIBUTION")
report.append("=" * 60)

for i, name in enumerate(CLASS_NAMES):
    report.append(
        f"{i:2d} | {name:<20} : {class_distribution[i]}"
    )

errors = (
    len(missing_labels)
    + len(missing_images)
    + len(empty_labels)
    + len(corrupted_images)
    + len(invalid_annotations)
    + len(invalid_class_ids)
)

report.append("")
report.append("=" * 60)

if errors == 0:
    report.append("✅ DATASET VERIFIED SUCCESSFULLY")
else:
    report.append(f"⚠ DATASET HAS {errors} ISSUES")

report.append("=" * 60)

# ============================================================
# PRINT REPORT
# ============================================================

for line in report:
    print(line)

# ============================================================
# SAVE REPORT
# ============================================================

report_file = REPORT_DIR / "dataset_verification_report.txt"

with open(report_file, "w", encoding="utf-8") as f:
    for line in report:
        f.write(line + "\n")

print(f"\nReport saved to:\n{report_file}")