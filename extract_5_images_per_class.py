import os
import shutil
import random
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

# Traffic sign dataset
DATASET_DIR = Path("datasets/traffic_sign")

# Output directory
OUTPUT_DIR = Path("traffic_sign_analysis")

# Number of images required for each class
IMAGES_PER_CLASS = 5

# Fixed seed so the same images are selected every time
random.seed(42)

# Traffic sign classes
CLASSES = {
    0: "regulatory",
    1: "mandatory",
    2: "informatory",
    3: "general",
    4: "warning"
}

# Dataset splits to search
SPLITS = ["train", "val", "test"]

# Supported image extensions
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"
}


# ============================================================
# FIND IMAGES FOR EACH CLASS
# ============================================================

class_images = {class_id: [] for class_id in CLASSES}

print("\nSearching traffic sign dataset...\n")

for split in SPLITS:

    image_dir = DATASET_DIR / "images" / split
    label_dir = DATASET_DIR / "labels" / split

    if not image_dir.exists():
        print(f"⚠ Image directory not found: {image_dir}")
        continue

    if not label_dir.exists():
        print(f"⚠ Label directory not found: {label_dir}")
        continue

    print(f"Scanning: {split}")

    for image_path in image_dir.iterdir():

        if image_path.suffix not in IMAGE_EXTENSIONS:
            continue

        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            continue

        # Read YOLO label file
        with open(label_path, "r") as f:
            lines = f.readlines()

        # Get all classes present in this image
        image_classes = set()

        for line in lines:

            parts = line.strip().split()

            if len(parts) < 5:
                continue

            class_id = int(parts[0])

            if class_id in CLASSES:
                image_classes.add(class_id)

        # Add image to every class it contains
        for class_id in image_classes:
            class_images[class_id].append(image_path)


# ============================================================
# COPY 5 IMAGES FOR EACH CLASS
# ============================================================

print("\nCreating analysis folders...\n")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for class_id, class_name in CLASSES.items():

    images = class_images[class_id]

    print(f"{class_name}: {len(images)} images found")

    # Shuffle so we don't always get the same first images
    random.shuffle(images)

    selected_images = images[:IMAGES_PER_CLASS]

    # Create class folder
    class_output_dir = OUTPUT_DIR / class_name
    class_output_dir.mkdir(parents=True, exist_ok=True)

    # Copy images
    for image_path in selected_images:

        destination = class_output_dir / image_path.name

        shutil.copy2(image_path, destination)

        print(f"  ✓ {image_path.name}")

    print(f"  → Copied {len(selected_images)} images\n")


print("=" * 60)
print("DONE")
print("=" * 60)

print(f"\nOutput directory:")
print(OUTPUT_DIR.resolve())