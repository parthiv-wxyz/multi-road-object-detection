import os
from pathlib import Path
from collections import Counter

# ============================================================
# CHANGE THIS PATH
# ============================================================
DATASET = Path(r"C:\Users\ppart\Desktop\YOLO\datasets\traffic_sign_classification_dataset")


# ============================================================
# HELPERS
# ============================================================
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(folder):
    if not folder.exists():
        return 0

    return sum(
        1
        for f in folder.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


def print_tree(path, prefix="", max_depth=3, depth=0):
    if depth > max_depth:
        return

    try:
        items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
    except PermissionError:
        return

    for i, item in enumerate(items):
        last = i == len(items) - 1
        connector = "└── " if last else "├── "

        print(prefix + connector + item.name)

        if item.is_dir():
            extension = "    " if last else "│   "
            print_tree(
                item,
                prefix + extension,
                max_depth,
                depth + 1
            )


# ============================================================
# CHECK DATASET
# ============================================================
print("=" * 70)
print("TRAFFIC SIGN DATASET CHECK")
print("=" * 70)

if not DATASET.exists():
    print(f"\nERROR: Dataset not found:")
    print(DATASET)
    print("\nChange the DATASET path in the script.")
    exit()

print(f"\nDataset path:")
print(DATASET)

print("\n" + "-" * 70)
print("DATASET STRUCTURE")
print("-" * 70)

print_tree(DATASET, max_depth=3)


# ============================================================
# CHECK FOR CLASSIFICATION FORMAT
# ============================================================
print("\n" + "-" * 70)
print("CLASSIFICATION FORMAT CHECK")
print("-" * 70)

splits = ["train", "val", "valid", "validation", "test"]

found_classification = False

for split in splits:
    split_path = DATASET / split

    if split_path.exists() and split_path.is_dir():

        class_dirs = [
            d for d in split_path.iterdir()
            if d.is_dir()
        ]

        if class_dirs:
            found_classification = True

            print(f"\n[{split.upper()}]")

            total = 0

            for class_dir in sorted(class_dirs):
                count = count_images(class_dir)
                total += count

                print(f"  {class_dir.name:<25} {count:>6} images")

            print(f"  {'TOTAL':<25} {total:>6} images")


# ============================================================
# CHECK FOR YOLO DETECTION FORMAT
# ============================================================
print("\n" + "-" * 70)
print("YOLO DETECTION FORMAT CHECK")
print("-" * 70)

images_dir = DATASET / "images"
labels_dir = DATASET / "labels"

if images_dir.exists():
    print("\nImages directory found:")
    print(images_dir)

    image_count = count_images(images_dir)
    print(f"Images: {image_count}")

if labels_dir.exists():
    print("\nLabels directory found:")
    print(labels_dir)

    txt_files = list(labels_dir.rglob("*.txt"))

    print(f"Label files: {len(txt_files)}")

    if txt_files:
        print("\nExample label files:")

        for txt in txt_files[:5]:
            print(f"  {txt.relative_to(DATASET)}")


# ============================================================
# SEARCH FOR YOLO TXT LABELS
# ============================================================
print("\n" + "-" * 70)
print("LABEL FORMAT CHECK")
print("-" * 70)

txt_files = list(DATASET.rglob("*.txt"))

if txt_files:
    print(f"\nFound {len(txt_files)} TXT files.")

    class_ids = Counter()

    for txt in txt_files:

        try:
            with open(txt, "r", encoding="utf-8") as f:
                for line in f:

                    line = line.strip()

                    if not line:
                        continue

                    parts = line.split()

                    # YOLO format:
                    # class x_center y_center width height
                    if len(parts) == 5:

                        try:
                            class_id = int(parts[0])
                            class_ids[class_id] += 1
                        except ValueError:
                            pass

        except Exception:
            pass

    if class_ids:
        print("\nYOLO class IDs found:")

        for class_id, count in sorted(class_ids.items()):
            print(f"  Class {class_id}: {count} objects")

else:
    print("\nNo TXT annotation files found.")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

if found_classification:
    print("""
CLASSIFICATION DATASET DETECTED

Your dataset appears to already have class folders.

Example:
train/
    regulatory/
    mandatory/
    ...

You can directly prepare it for YOLOv5 classification.
""")

elif labels_dir.exists() or txt_files:
    print("""
YOLO DETECTION DATASET DETECTED

Your dataset appears to contain bounding-box annotations.

We can convert/crop the detected signs and create a
classification dataset from it.
""")

else:
    print("""
UNKNOWN DATASET FORMAT

The dataset does not appear to be in a standard
classification or YOLO detection structure.

We need to inspect its files manually.
""")

print("=" * 70)
print("CHECK COMPLETE")
print("=" * 70)