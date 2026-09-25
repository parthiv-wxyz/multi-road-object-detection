from pathlib import Path
from collections import Counter

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "datasets" / "traffic_sign"

CLASS_NAMES = {
    0: "regulatory",
    1: "mandatory",
    2: "informatory",
    3: "general",
    4: "warning",
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


# ============================================================
# VERIFY SPLIT
# ============================================================

def verify_split(split):

    images_dir = DATASET_ROOT / "images" / split
    labels_dir = DATASET_ROOT / "labels" / split

    print("\n" + "=" * 65)
    print(f"{split.upper()} SPLIT")
    print("=" * 65)

    images = [
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    labels = list(labels_dir.glob("*.txt"))

    print(f"Images : {len(images)}")
    print(f"Labels : {len(labels)}")

    # --------------------------------------------------------
    # Image-label matching
    # --------------------------------------------------------

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems

    print("\nImage/Label matching:")

    if not missing_labels:
        print("[OK] All images have labels.")
    else:
        print(f"[ERROR] {len(missing_labels)} images have no labels.")

    if not missing_images:
        print("[OK] All labels have images.")
    else:
        print(f"[ERROR] {len(missing_images)} labels have no images.")

    # --------------------------------------------------------
    # Annotation analysis
    # --------------------------------------------------------

    class_counter = Counter()

    invalid_boxes = 0
    invalid_classes = 0
    invalid_format = 0
    empty_labels = 0
    total_objects = 0

    for label_file in labels:

        with open(label_file, "r", encoding="utf-8") as f:
            lines = [
                line.strip()
                for line in f
                if line.strip()
            ]

        if not lines:
            empty_labels += 1
            continue

        for line in lines:

            parts = line.split()

            if len(parts) != 5:
                invalid_format += 1
                continue

            try:
                class_id = int(float(parts[0]))

                x = float(parts[1])
                y = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])

            except ValueError:
                invalid_format += 1
                continue

            if class_id not in CLASS_NAMES:
                invalid_classes += 1
                continue

            class_counter[class_id] += 1
            total_objects += 1

            if not (
                0 <= x <= 1
                and 0 <= y <= 1
                and 0 < w <= 1
                and 0 < h <= 1
            ):
                invalid_boxes += 1

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print("\nClass distribution:")

    for class_id, class_name in CLASS_NAMES.items():

        count = class_counter[class_id]

        percentage = (
            count / total_objects * 100
            if total_objects
            else 0
        )

        print(
            f"{class_id}: {class_name:<12}"
            f"{count:>5} objects "
            f"({percentage:6.2f}%)"
        )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("\nAnnotation validation:")

    print(f"Total objects       : {total_objects}")
    print(f"Empty labels        : {empty_labels}")
    print(f"Invalid format      : {invalid_format}")
    print(f"Invalid classes     : {invalid_classes}")
    print(f"Invalid boxes       : {invalid_boxes}")

    # Check whether every class exists
    missing_classes = [
        CLASS_NAMES[i]
        for i in CLASS_NAMES
        if class_counter[i] == 0
    ]

    if missing_classes:
        print(
            "\n[WARNING] Classes missing from this split:"
        )

        for name in missing_classes:
            print(f"  - {name}")

    else:
        print("\n[OK] All 5 classes are represented.")

    return {
        "images": len(images),
        "labels": len(labels),
        "objects": total_objects,
        "classes": class_counter,
        "invalid": (
            invalid_boxes
            + invalid_classes
            + invalid_format
            + empty_labels
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "#" * 65)
    print("#       TRAFFIC SIGN TRAIN/VAL/TEST VERIFICATION")
    print("#" * 65)

    results = {}

    for split in ["train", "val", "test"]:
        results[split] = verify_split(split)

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    print("\n\n" + "=" * 65)
    print("OVERALL DATASET SUMMARY")
    print("=" * 65)

    total_images = 0
    total_labels = 0
    total_objects = 0
    total_invalid = 0

    for split, result in results.items():

        total_images += result["images"]
        total_labels += result["labels"]
        total_objects += result["objects"]
        total_invalid += result["invalid"]

        print(
            f"{split.upper():5} | "
            f"Images: {result['images']:3} | "
            f"Labels: {result['labels']:3} | "
            f"Objects: {result['objects']:4}"
        )

    print("-" * 65)

    print(f"TOTAL | Images: {total_images:3}")
    print(f"TOTAL | Labels: {total_labels:3}")
    print(f"TOTAL | Objects: {total_objects:4}")
    print(f"Invalid annotations: {total_invalid}")

    print("\n" + "=" * 65)

    if total_invalid == 0:
        print("[RESULT] SPLIT PASSED VALIDATION.")
    else:
        print("[RESULT] SPLIT NEEDS ATTENTION.")

    print("=" * 65)


if __name__ == "__main__":
    main()