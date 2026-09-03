from pathlib import Path
from collections import Counter
import shutil

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = Path(
    r"E:\Parthiv\multi-road-object-detection\datasets\idd"
)

SPLITS = ["train", "val"]

NUM_CLASSES = 7

# Boxes with almost zero area after clipping will be removed
MIN_WIDTH = 0.0001
MIN_HEIGHT = 0.0001


# ============================================================
# STATISTICS
# ============================================================

stats = Counter()

modified_files = []


# ============================================================
# CLEAN A SINGLE LABEL FILE
# ============================================================

def clean_label_file(label_path):

    try:
        lines = label_path.read_text().splitlines()
    except Exception as e:
        print(f"Cannot read: {label_path}")
        print(e)
        return

    cleaned_lines = []
    seen = set()
    file_modified = False

    for line_number, line in enumerate(lines, start=1):

        line = line.strip()

        # Ignore blank lines
        if not line:
            file_modified = True
            continue

        parts = line.split()

        # Invalid format
        if len(parts) != 5:
            stats["invalid_format_removed"] += 1
            file_modified = True
            continue

        try:
            class_id = int(parts[0])

            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

        except ValueError:
            stats["non_numeric_removed"] += 1
            file_modified = True
            continue

        # Invalid class ID
        if class_id < 0 or class_id >= NUM_CLASSES:
            stats["invalid_class_removed"] += 1
            file_modified = True
            continue

        # Invalid dimensions
        if width <= 0 or height <= 0:
            stats["invalid_size_removed"] += 1
            file_modified = True
            continue

        # ====================================================
        # Convert YOLO coordinates to corner coordinates
        # ====================================================

        x1 = x_center - width / 2
        y1 = y_center - height / 2

        x2 = x_center + width / 2
        y2 = y_center + height / 2

        original_x1 = x1
        original_y1 = y1
        original_x2 = x2
        original_y2 = y2

        # ====================================================
        # CLIP BOX TO IMAGE BOUNDARIES
        # ====================================================

        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))

        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))

        # Was clipping necessary?
        if (
            original_x1 != x1
            or original_y1 != y1
            or original_x2 != x2
            or original_y2 != y2
        ):
            stats["boxes_clipped"] += 1
            file_modified = True

        # ====================================================
        # Calculate new YOLO coordinates
        # ====================================================

        new_width = x2 - x1
        new_height = y2 - y1

        # Remove boxes that disappear after clipping
        if (
            new_width < MIN_WIDTH
            or new_height < MIN_HEIGHT
        ):
            stats["invalid_boxes_removed"] += 1
            file_modified = True
            continue

        new_x_center = (x1 + x2) / 2
        new_y_center = (y1 + y2) / 2

        # Create standardized annotation
        annotation = (
            class_id,
            round(new_x_center, 6),
            round(new_y_center, 6),
            round(new_width, 6),
            round(new_height, 6)
        )

        # ====================================================
        # REMOVE DUPLICATES
        # ====================================================

        if annotation in seen:
            stats["duplicates_removed"] += 1
            file_modified = True
            continue

        seen.add(annotation)

        cleaned_line = (
            f"{class_id} "
            f"{new_x_center:.6f} "
            f"{new_y_center:.6f} "
            f"{new_width:.6f} "
            f"{new_height:.6f}"
        )

        cleaned_lines.append(cleaned_line)

    # ========================================================
    # SAVE ONLY IF CHANGES WERE MADE
    # ========================================================

    if file_modified:

        # Backup original label
        backup_path = label_path.with_suffix(".txt.backup")

        if not backup_path.exists():
            shutil.copy2(label_path, backup_path)

        # Write cleaned label
        with open(label_path, "w") as f:

            if cleaned_lines:
                f.write("\n".join(cleaned_lines) + "\n")

        modified_files.append(label_path)

        stats["modified_files"] += 1


# ============================================================
# PROCESS DATASET
# ============================================================

print("=" * 65)
print("IDD DATASET CLEANING")
print("=" * 65)

for split in SPLITS:

    labels_dir = DATASET_PATH / split / "labels"

    print(f"\nProcessing {split.upper()} dataset...")
    print(f"Path: {labels_dir}")

    if not labels_dir.exists():

        print(f"ERROR: Folder does not exist: {labels_dir}")
        continue

    label_files = list(labels_dir.glob("*.txt"))

    print(f"Label files found: {len(label_files)}")

    for index, label_path in enumerate(label_files, start=1):

        clean_label_file(label_path)

        if index % 5000 == 0:

            print(
                f"Processed {index}/{len(label_files)} files"
            )


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 65)
print("CLEANING REPORT")
print("=" * 65)

print(
    f"\nFiles Modified: {stats['modified_files']}"
)

print(
    f"Bounding Boxes Clipped: {stats['boxes_clipped']}"
)

print(
    f"Duplicate Annotations Removed: "
    f"{stats['duplicates_removed']}"
)

print(
    f"Invalid Boxes Removed: "
    f"{stats['invalid_boxes_removed']}"
)

print(
    f"Invalid Format Removed: "
    f"{stats['invalid_format_removed']}"
)

print(
    f"Invalid Class IDs Removed: "
    f"{stats['invalid_class_removed']}"
)

print(
    f"Invalid Sizes Removed: "
    f"{stats['invalid_size_removed']}"
)

print("\n" + "=" * 65)
print("DATASET CLEANING COMPLETED")
print("=" * 65)

print(
    "\nIMPORTANT: Run check_dataset.py again to verify "
    "the cleaned dataset."
)