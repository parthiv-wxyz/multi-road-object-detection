from pathlib import Path
from PIL import Image
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = Path(r"E:\Parthiv\multi-road-object-detection\datasets\idd")

CLASSES = [
    "person",
    "car",
    "autorickshaw",
    "truck",
    "bus",
    "motorcycle",
    "bicycle"
]

NUM_CLASSES = len(CLASSES)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}

# ============================================================
# COUNTERS
# ============================================================

stats = Counter()

missing_labels = []
missing_images = []
corrupted_images = []

invalid_labels = []
invalid_class_ids = []
invalid_boxes = []

duplicate_annotations = []
empty_labels = []

class_distribution = Counter()


# ============================================================
# CHECK LABEL FILE
# ============================================================

def check_label(label_path):

    annotations = []

    try:
        with open(label_path, "r") as f:
            lines = f.readlines()

    except Exception as e:
        invalid_labels.append(
            (label_path, f"Cannot read file: {e}")
        )
        return

    # Empty label file
    if not lines or all(not line.strip() for line in lines):
        empty_labels.append(label_path)
        return

    seen_annotations = set()

    for line_number, line in enumerate(lines, start=1):

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        # Check number of values
        if len(parts) != 5:

            invalid_labels.append(
                (
                    label_path,
                    f"Line {line_number}: Expected 5 values, found {len(parts)}"
                )
            )

            continue

        # Convert to numbers
        try:

            class_id = int(parts[0])

            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

        except ValueError:

            invalid_labels.append(
                (
                    label_path,
                    f"Line {line_number}: Non-numeric values"
                )
            )

            continue

        # Check class ID
        if class_id < 0 or class_id >= NUM_CLASSES:

            invalid_class_ids.append(
                (
                    label_path,
                    f"Line {line_number}: Invalid class ID {class_id}"
                )
            )

            continue

        # Check normalized values
        if not (
            0 <= x_center <= 1 and
            0 <= y_center <= 1 and
            0 < width <= 1 and
            0 < height <= 1
        ):

            invalid_boxes.append(
                (
                    label_path,
                    f"Line {line_number}: Values outside valid range"
                )
            )

            continue

        # Check if bounding box goes outside image
        x_min = x_center - width / 2
        x_max = x_center + width / 2

        y_min = y_center - height / 2
        y_max = y_center + height / 2

        if (
            x_min < 0 or
            x_max > 1 or
            y_min < 0 or
            y_max > 1
        ):

            invalid_boxes.append(
                (
                    label_path,
                    f"Line {line_number}: Bounding box exceeds image boundary"
                )
            )

        # Check duplicate annotations
        annotation = tuple(parts)

        if annotation in seen_annotations:

            duplicate_annotations.append(
                (
                    label_path,
                    f"Line {line_number}: Duplicate annotation"
                )
            )

        seen_annotations.add(annotation)

        # Count class
        class_distribution[class_id] += 1


# ============================================================
# CHECK DATASET SPLIT
# ============================================================

def check_split(split):

    print("\n" + "=" * 60)
    print(f"CHECKING {split.upper()} DATASET")
    print("=" * 60)

    images_dir = DATASET_PATH / split / "images"
    labels_dir = DATASET_PATH / split / "labels"

    if not images_dir.exists():
        print(f"\nERROR: Images folder not found:\n{images_dir}")
        return

    if not labels_dir.exists():
        print(f"\nERROR: Labels folder not found:\n{labels_dir}")
        return

    # Get images
    image_files = [
        f for f in images_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    # Get labels
    label_files = list(labels_dir.glob("*.txt"))

    print(f"\nImages found: {len(image_files)}")
    print(f"Labels found: {len(label_files)}")

    stats[f"{split}_images"] = len(image_files)
    stats[f"{split}_labels"] = len(label_files)

    image_stems = set()
    label_stems = set()

    # Check images
    for image_path in image_files:

        image_stems.add(image_path.stem)

        try:
            with Image.open(image_path) as img:
                img.verify()

        except Exception as e:

            corrupted_images.append(
                (image_path, str(e))
            )

        # Check corresponding label
        label_path = labels_dir / f"{image_path.stem}.txt"

        if not label_path.exists():

            missing_labels.append(image_path)

    # Check labels
    for label_path in label_files:

        label_stems.add(label_path.stem)

        # Check corresponding image
        possible_images = [
            images_dir / f"{label_path.stem}.jpg",
            images_dir / f"{label_path.stem}.JPG",
            images_dir / f"{label_path.stem}.jpeg",
            images_dir / f"{label_path.stem}.JPEG",
            images_dir / f"{label_path.stem}.png",
            images_dir / f"{label_path.stem}.PNG",
        ]

        if not any(image.exists() for image in possible_images):

            missing_images.append(label_path)

        # Check label content
        check_label(label_path)


# ============================================================
# RUN CHECKS
# ============================================================

print("=" * 60)
print("YOLO DATASET VALIDATION")
print("=" * 60)

print(f"\nDataset Path:\n{DATASET_PATH}")

if not DATASET_PATH.exists():

    print("\nERROR: Dataset path does not exist.")
    print("Please update DATASET_PATH in the script.")

    exit()


# Check train and validation datasets
check_split("train")
check_split("val")


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n\n")
print("=" * 60)
print("FINAL DATASET REPORT")
print("=" * 60)

print("\nDATASET SIZE")

print(
    f"Train Images : {stats['train_images']}"
)

print(
    f"Train Labels : {stats['train_labels']}"
)

print(
    f"Validation Images : {stats['val_images']}"
)

print(
    f"Validation Labels : {stats['val_labels']}"
)


print("\n" + "-" * 60)
print("ERROR SUMMARY")
print("-" * 60)

print(f"Missing Labels          : {len(missing_labels)}")
print(f"Missing Images          : {len(missing_images)}")
print(f"Corrupted Images        : {len(corrupted_images)}")
print(f"Invalid Label Lines     : {len(invalid_labels)}")
print(f"Invalid Class IDs       : {len(invalid_class_ids)}")
print(f"Invalid Bounding Boxes  : {len(invalid_boxes)}")
print(f"Duplicate Annotations   : {len(duplicate_annotations)}")
print(f"Empty Label Files       : {len(empty_labels)}")


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "-" * 60)
print("CLASS DISTRIBUTION")
print("-" * 60)

for class_id, class_name in enumerate(CLASSES):

    count = class_distribution[class_id]

    print(
        f"{class_id} - {class_name}: {count}"
    )


# ============================================================
# PRINT ERROR DETAILS
# ============================================================

def print_errors(title, errors, limit=20):

    if not errors:
        return

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    for error in errors[:limit]:

        print(error)

    if len(errors) > limit:

        print(
            f"\n... and {len(errors) - limit} more"
        )


print_errors(
    "MISSING LABELS",
    missing_labels
)

print_errors(
    "MISSING IMAGES",
    missing_images
)

print_errors(
    "CORRUPTED IMAGES",
    corrupted_images
)

print_errors(
    "INVALID LABELS",
    invalid_labels
)

print_errors(
    "INVALID CLASS IDs",
    invalid_class_ids
)

print_errors(
    "INVALID BOUNDING BOXES",
    invalid_boxes
)

print_errors(
    "DUPLICATE ANNOTATIONS",
    duplicate_annotations
)


# ============================================================
# FINAL STATUS
# ============================================================

total_errors = (
    len(missing_labels)
    + len(missing_images)
    + len(corrupted_images)
    + len(invalid_labels)
    + len(invalid_class_ids)
    + len(invalid_boxes)
)

print("\n" + "=" * 60)

if total_errors == 0:

    print("DATASET VALIDATION PASSED")

else:

    print(f"DATASET VALIDATION FOUND {total_errors} POTENTIAL ERRORS")

print("=" * 60)