import json
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "datasets" / "road_damage"

ANNOTATIONS_DIR = DATASET_ROOT / "annotations"
LABELS_DIR = DATASET_ROOT / "labels"

SPLITS = {
    "train": "voc07_train.json",
    "val": "voc07_trainval.json",
    "test": "voc07_test.json",
}

# ============================================================
# CLASSES
# ============================================================

EXPECTED_CLASSES = {
    0: "Alligator",
    1: "Block",
    2: "Longitudinal",
    3: "Transversal",
    4: "Pot Hole",
}

# ============================================================
# CONVERSION
# ============================================================

total_images = 0
total_annotations = 0
total_converted = 0
total_clipped = 0
total_removed = 0

for split, json_name in SPLITS.items():

    json_path = ANNOTATIONS_DIR / json_name
    output_dir = LABELS_DIR / split

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"CONVERTING {split.upper()}")
    print("=" * 70)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]
    categories = data["categories"]

    # --------------------------------------------------------
    # Verify classes
    # --------------------------------------------------------

    actual_classes = {
        category["id"]: category["name"]
        for category in categories
    }

    if actual_classes != EXPECTED_CLASSES:
        raise ValueError(
            f"Unexpected class mapping in {json_name}:\n"
            f"{actual_classes}"
        )

    # --------------------------------------------------------
    # Image lookup
    # --------------------------------------------------------

    image_map = {
        image["id"]: image
        for image in images
    }

    # --------------------------------------------------------
    # Group annotations by image
    # --------------------------------------------------------

    annotations_by_image = {}

    for annotation in annotations:
        image_id = annotation["image_id"]

        annotations_by_image.setdefault(
            image_id, []
        ).append(annotation)

    split_converted = 0
    split_clipped = 0
    split_removed = 0

    # --------------------------------------------------------
    # Process each image
    # --------------------------------------------------------

    for image in images:

        image_id = image["id"]
        file_name = image["file_name"]

        image_width = image["width"]
        image_height = image["height"]

        label_name = Path(file_name).stem + ".txt"
        label_path = output_dir / label_name

        yolo_lines = []

        for annotation in annotations_by_image.get(
            image_id, []
        ):

            category_id = annotation["category_id"]

            if category_id not in EXPECTED_CLASSES:
                raise ValueError(
                    f"Unknown category ID {category_id} "
                    f"in annotation {annotation['id']}"
                )

            x, y, w, h = annotation["bbox"]

            # ------------------------------------------------
            # Remove genuinely unusable boxes
            # ------------------------------------------------

            if w <= 0 or h <= 0:
                split_removed += 1
                continue

            # ------------------------------------------------
            # Clip bbox to image boundaries
            # ------------------------------------------------

            original_x = x
            original_y = y
            original_w = w
            original_h = h

            x1 = max(0.0, x)
            y1 = max(0.0, y)

            x2 = min(
                float(image_width),
                x + w
            )

            y2 = min(
                float(image_height),
                y + h
            )

            clipped_w = x2 - x1
            clipped_h = y2 - y1

            # Safety check
            if clipped_w <= 0 or clipped_h <= 0:
                split_removed += 1
                continue

            if (
                x1 != original_x
                or y1 != original_y
                or clipped_w != original_w
                or clipped_h != original_h
            ):
                split_clipped += 1

            # ------------------------------------------------
            # COCO → YOLO
            # ------------------------------------------------

            x_center = (x1 + x2) / 2.0
            y_center = (y1 + y2) / 2.0

            box_width = clipped_w
            box_height = clipped_h

            # Normalize
            x_center /= image_width
            y_center /= image_height
            box_width /= image_width
            box_height /= image_height

            yolo_lines.append(
                f"{category_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{box_width:.6f} "
                f"{box_height:.6f}"
            )

            split_converted += 1

        # ----------------------------------------------------
        # Write label
        # ----------------------------------------------------

        with open(label_path, "w", encoding="utf-8") as f:
            if yolo_lines:
                f.write("\n".join(yolo_lines))
                f.write("\n")

    # --------------------------------------------------------
    # Split summary
    # --------------------------------------------------------

    print(f"Images              : {len(images)}")
    print(f"Original annotations: {len(annotations)}")
    print(f"Converted boxes     : {split_converted}")
    print(f"Clipped boxes       : {split_clipped}")
    print(f"Removed boxes       : {split_removed}")
    print(f"Labels directory    : {output_dir}")

    total_images += len(images)
    total_annotations += len(annotations)
    total_converted += split_converted
    total_clipped += split_clipped
    total_removed += split_removed

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("RDD4D CONVERSION COMPLETE")
print("=" * 70)

print(f"Total images         : {total_images}")
print(f"Total annotations    : {total_annotations}")
print(f"Total YOLO boxes     : {total_converted}")
print(f"Total clipped boxes  : {total_clipped}")
print(f"Total removed boxes  : {total_removed}")

print("\nClass mapping:")

for class_id, class_name in EXPECTED_CLASSES.items():
    print(f"  {class_id} -> {class_name}")

print("=" * 70)