from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "road_damage"

SPLITS = ["train", "val", "test"]

CLASS_NAMES = {
    0: "Alligator",
    1: "Block",
    2: "Longitudinal",
    3: "Transversal",
    4: "Pot Hole",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"}

total_images = 0
total_labels = 0
total_boxes = 0

global_class_counts = Counter()

print("=" * 70)
print("RDD4D YOLO DATASET VERIFICATION")
print("=" * 70)

for split in SPLITS:

    image_dir = DATASET / "images" / split
    label_dir = DATASET / "labels" / split

    images = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix in IMAGE_EXTENSIONS
    ]

    labels = list(label_dir.glob("*.txt"))

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing_labels = image_stems - label_stems
    extra_labels = label_stems - image_stems

    class_counts = Counter()

    invalid_boxes = []
    empty_labels = 0
    boxes = 0

    for label_file in labels:

        with open(label_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            empty_labels += 1
            continue

        for line_number, line in enumerate(lines, 1):

            parts = line.split()

            if len(parts) != 5:
                invalid_boxes.append(
                    (label_file.name, line_number, "wrong field count")
                )
                continue

            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
            except ValueError:
                invalid_boxes.append(
                    (label_file.name, line_number, "non-numeric value")
                )
                continue

            if class_id not in CLASS_NAMES:
                invalid_boxes.append(
                    (label_file.name, line_number, "invalid class")
                )
                continue

            if not (
                0 <= x_center <= 1
                and 0 <= y_center <= 1
                and 0 < width <= 1
                and 0 < height <= 1
            ):
                invalid_boxes.append(
                    (label_file.name, line_number, "coordinate out of range")
                )
                continue

            class_counts[class_id] += 1
            global_class_counts[class_id] += 1
            boxes += 1

    print("\n" + "-" * 70)
    print(split.upper())
    print("-" * 70)

    print(f"Images             : {len(images)}")
    print(f"Labels             : {len(labels)}")
    print(f"Bounding boxes     : {boxes}")
    print(f"Empty labels       : {empty_labels}")
    print(f"Missing labels     : {len(missing_labels)}")
    print(f"Extra labels       : {len(extra_labels)}")
    print(f"Invalid boxes      : {len(invalid_boxes)}")

    print("\nClass distribution:")

    for class_id, name in CLASS_NAMES.items():
        print(
            f"  {class_id} -> {name:15} : "
            f"{class_counts[class_id]}"
        )

    if missing_labels:
        print("\nMissing label examples:")
        for name in sorted(missing_labels)[:10]:
            print(" ", name)

    if extra_labels:
        print("\nExtra label examples:")
        for name in sorted(extra_labels)[:10]:
            print(" ", name)

    if invalid_boxes:
        print("\nInvalid box examples:")
        for item in invalid_boxes[:10]:
            print(" ", item)

    total_images += len(images)
    total_labels += len(labels)
    total_boxes += boxes


print("\n" + "=" * 70)
print("GLOBAL SUMMARY")
print("=" * 70)

print(f"Images       : {total_images}")
print(f"Labels       : {total_labels}")
print(f"Boxes        : {total_boxes}")

print("\nAll-class distribution:")

for class_id, name in CLASS_NAMES.items():
    print(
        f"  {class_id} -> {name:15} : "
        f"{global_class_counts[class_id]}"
    )

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)