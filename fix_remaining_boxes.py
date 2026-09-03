from pathlib import Path

DATASET_PATH = Path(
    r"E:\Parthiv\multi-road-object-detection\datasets\idd"
)

SPLITS = ["train", "val"]

fixed_boxes = 0
removed_boxes = 0
modified_files = 0


def fix_label_file(label_path):

    global fixed_boxes
    global removed_boxes
    global modified_files

    lines = label_path.read_text().splitlines()

    new_lines = []
    modified = False

    for line in lines:

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            continue

        try:
            class_id = int(parts[0])

            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

        except ValueError:
            continue

        # Convert YOLO to corners
        x1 = xc - w / 2
        y1 = yc - h / 2

        x2 = xc + w / 2
        y2 = yc + h / 2

        original = (x1, y1, x2, y2)

        # Strict clipping
        x1 = max(0.0, min(x1, 1.0))
        y1 = max(0.0, min(y1, 1.0))

        x2 = max(0.0, min(x2, 1.0))
        y2 = max(0.0, min(y2, 1.0))

        # Calculate new dimensions
        w = x2 - x1
        h = y2 - y1

        # Remove unusable boxes
        if w <= 0 or h <= 0:

            removed_boxes += 1
            modified = True

            continue

        # Convert corners back to YOLO
        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2

        # Final safety adjustment
        xc = min(max(xc, 0.0), 1.0)
        yc = min(max(yc, 0.0), 1.0)

        # Check whether modification occurred
        if original != (x1, y1, x2, y2):

            fixed_boxes += 1
            modified = True

        # Use high precision to avoid unnecessary rounding errors
        new_lines.append(
            f"{class_id} "
            f"{xc:.10f} "
            f"{yc:.10f} "
            f"{w:.10f} "
            f"{h:.10f}"
        )

    if modified:

        label_path.write_text(
            "\n".join(new_lines) + "\n"
            if new_lines else ""
        )

        modified_files += 1


print("=" * 60)
print("FIXING REMAINING IDD BOUNDING BOXES")
print("=" * 60)

for split in SPLITS:

    labels_dir = DATASET_PATH / split / "labels"

    print(f"\nProcessing {split}...")

    files = list(labels_dir.glob("*.txt"))

    for i, label_file in enumerate(files, 1):

        fix_label_file(label_file)

        if i % 5000 == 0:

            print(
                f"Processed {i}/{len(files)} files"
            )


print("\n" + "=" * 60)
print("FINAL FIX REPORT")
print("=" * 60)

print(f"Bounding boxes fixed: {fixed_boxes}")
print(f"Bounding boxes removed: {removed_boxes}")
print(f"Files modified: {modified_files}")

print("\nCompleted.")