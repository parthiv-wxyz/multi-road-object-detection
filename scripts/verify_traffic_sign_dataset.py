from pathlib import Path
from collections import Counter
from PIL import Image

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "datasets" / "traffic_sign"

IMAGES_DIR = DATASET_ROOT / "images"
LABELS_DIR = DATASET_ROOT / "labels"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

CLASS_NAMES = {
    0: "regulatory",
    1: "mandatory",
    2: "informatory",
    3: "general",
    4: "warning",
}


# ============================================================
# MAIN AUDIT
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("             TRAFFIC SIGN DATASET AUDIT")
    print("=" * 70)

    print(f"\nDataset:")
    print(DATASET_ROOT)

    # --------------------------------------------------------
    # Check directories
    # --------------------------------------------------------

    if not DATASET_ROOT.exists():
        print("\n[ERROR] Dataset folder does not exist.")
        return

    if not IMAGES_DIR.exists():
        print("\n[ERROR] images folder does not exist.")
        return

    if not LABELS_DIR.exists():
        print("\n[ERROR] labels folder does not exist.")
        return

    # --------------------------------------------------------
    # Find images and labels
    # --------------------------------------------------------

    images = [
        p for p in IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    labels = list(LABELS_DIR.glob("*.txt"))

    print("\n" + "-" * 70)
    print("FILE COUNTS")
    print("-" * 70)

    print(f"Images : {len(images)}")
    print(f"Labels : {len(labels)}")

    # --------------------------------------------------------
    # Match images and labels
    # --------------------------------------------------------

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems

    print("\n" + "-" * 70)
    print("IMAGE / LABEL MATCHING")
    print("-" * 70)

    if not missing_labels:
        print("[OK] Every image has a corresponding label.")
    else:
        print(
            f"[WARNING] {len(missing_labels)} images "
            f"have no label."
        )

        for name in sorted(missing_labels)[:10]:
            print("   ", name)

    if not missing_images:
        print("[OK] Every label has a corresponding image.")
    else:
        print(
            f"[WARNING] {len(missing_images)} labels "
            f"have no image."
        )

        for name in sorted(missing_images)[:10]:
            print("   ", name)

    # --------------------------------------------------------
    # Analyze labels
    # --------------------------------------------------------

    class_counter = Counter()

    invalid_boxes = 0
    invalid_classes = 0
    invalid_format = 0
    empty_labels = 0

    total_boxes = 0

    print("\n" + "-" * 70)
    print("ANNOTATION CHECK")
    print("-" * 70)

    for label_file in labels:

        try:

            with open(
                label_file,
                "r",
                encoding="utf-8"
            ) as f:

                lines = [
                    line.strip()
                    for line in f
                    if line.strip()
                ]

        except Exception as e:

            print(
                f"[ERROR] Cannot read "
                f"{label_file.name}: {e}"
            )

            continue

        # Empty label
        if not lines:

            empty_labels += 1
            continue

        for line_number, line in enumerate(
            lines,
            start=1
        ):

            parts = line.split()

            # YOLO format:
            # class x_center y_center width height

            if len(parts) != 5:

                invalid_format += 1

                print(
                    f"[INVALID FORMAT] "
                    f"{label_file.name} "
                    f"line {line_number}"
                )

                continue

            try:

                class_id = int(float(parts[0]))

                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

            except ValueError:

                invalid_format += 1

                print(
                    f"[INVALID VALUE] "
                    f"{label_file.name} "
                    f"line {line_number}"
                )

                continue

            # ------------------------------------------------
            # Class ID
            # ------------------------------------------------

            if class_id not in CLASS_NAMES:

                invalid_classes += 1

                print(
                    f"[INVALID CLASS] "
                    f"{label_file.name}: "
                    f"class={class_id}"
                )

                continue

            class_counter[class_id] += 1
            total_boxes += 1

            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            valid_box = (
                0 <= x_center <= 1
                and
                0 <= y_center <= 1
                and
                0 < width <= 1
                and
                0 < height <= 1
            )

            if not valid_box:

                invalid_boxes += 1

                print(
                    f"[INVALID BOX] "
                    f"{label_file.name}: "
                    f"{line}"
                )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("CLASS DISTRIBUTION")
    print("-" * 70)

    for class_id, class_name in CLASS_NAMES.items():

        count = class_counter[class_id]

        percentage = (
            count / total_boxes * 100
            if total_boxes > 0
            else 0
        )

        print(
            f"{class_id} = {class_name:<12} "
            f"{count:>6} objects "
            f"({percentage:6.2f}%)"
        )

    # --------------------------------------------------------
    # Image information
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("IMAGE INFORMATION")
    print("-" * 70)

    widths = []
    heights = []
    unreadable = 0

    for image_file in images:

        try:

            with Image.open(image_file) as img:

                widths.append(img.width)
                heights.append(img.height)

        except Exception:

            unreadable += 1

    if widths:

        print(
            f"Width range  : "
            f"{min(widths)} - {max(widths)}"
        )

        print(
            f"Height range : "
            f"{min(heights)} - {max(heights)}"
        )

        print(
            f"Unreadable images : "
            f"{unreadable}"
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"Images                : {len(images)}")
    print(f"Labels                : {len(labels)}")
    print(f"Total objects         : {total_boxes}")
    print(f"Missing labels        : {len(missing_labels)}")
    print(f"Labels without images : {len(missing_images)}")
    print(f"Empty labels          : {empty_labels}")
    print(f"Invalid format        : {invalid_format}")
    print(f"Invalid classes       : {invalid_classes}")
    print(f"Invalid boxes         : {invalid_boxes}")

    if (
        not missing_labels
        and
        not missing_images
        and
        empty_labels == 0
        and
        invalid_format == 0
        and
        invalid_classes == 0
        and
        invalid_boxes == 0
    ):

        print("\n[RESULT] DATASET PASSED BASIC YOLO VALIDATION.")

    else:

        print(
            "\n[RESULT] DATASET NEEDS CLEANING "
            "BEFORE TRAINING."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()