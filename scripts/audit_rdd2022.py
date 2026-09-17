from pathlib import Path
from collections import Counter

DATASET = Path(r"E:\Parthiv\multi-road-object-detection\datasets\rdd")

CLASSES = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Other Corruption",
    4: "Pothole",
}

print("=" * 70)
print("RDD2022 DATASET AUDIT")
print("=" * 70)

for split in ["train", "val", "test"]:
    print(f"\n{'-' * 70}")
    print(split.upper())
    print("-" * 70)

    image_dir = DATASET / split / "images"
    label_dir = DATASET / split / "labels"

    image_files = [
        f for f in image_dir.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    label_files = list(label_dir.glob("*.txt"))

    print(f"Images : {len(image_files)}")
    print(f"Labels : {len(label_files)}")

    image_stems = {f.stem for f in image_files}
    label_stems = {f.stem for f in label_files}

    missing_labels = image_stems - label_stems
    extra_labels = label_stems - image_stems

    print(f"Missing labels : {len(missing_labels)}")
    print(f"Extra labels   : {len(extra_labels)}")

    class_counts = Counter()
    total_boxes = 0
    empty_labels = 0
    invalid_boxes = []

    for label_file in label_files:

        lines = label_file.read_text(
            encoding="utf-8",
            errors="ignore"
        ).strip().splitlines()

        if not lines:
            empty_labels += 1
            continue

        for line_number, line in enumerate(lines, start=1):
            parts = line.split()

            if len(parts) != 5:
                invalid_boxes.append(
                    (label_file.name, line_number, "invalid format")
                )
                continue

            try:
                class_id = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                invalid_boxes.append(
                    (label_file.name, line_number, "non-numeric value")
                )
                continue

            if class_id not in CLASSES:
                invalid_boxes.append(
                    (label_file.name, line_number, f"invalid class {class_id}")
                )
                continue

            if not (
                0 <= x <= 1 and
                0 <= y <= 1 and
                0 < w <= 1 and
                0 < h <= 1
            ):
                invalid_boxes.append(
                    (label_file.name, line_number, "invalid coordinates")
                )
                continue

            class_counts[class_id] += 1
            total_boxes += 1

    print(f"Empty labels   : {empty_labels}")
    print(f"Bounding boxes : {total_boxes}")
    print(f"Invalid boxes  : {len(invalid_boxes)}")

    print("\nClass distribution:")

    for class_id, class_name in CLASSES.items():
        print(
            f"  {class_id} -> "
            f"{class_name:<22}: "
            f"{class_counts[class_id]}"
        )

    if invalid_boxes:
        print("\nInvalid examples:")
        for example in invalid_boxes[:10]:
            print(f"  {example}")

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)