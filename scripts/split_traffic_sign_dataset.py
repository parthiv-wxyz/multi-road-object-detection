from pathlib import Path
import random
import shutil

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "datasets" / "traffic_sign"

SOURCE_IMAGES = DATASET_ROOT / "images"
SOURCE_LABELS = DATASET_ROOT / "labels"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

# ============================================================
# CREATE DIRECTORIES
# ============================================================

splits = ["train", "val", "test"]

for split in splits:
    (SOURCE_IMAGES / split).mkdir(parents=True, exist_ok=True)
    (SOURCE_LABELS / split).mkdir(parents=True, exist_ok=True)


# ============================================================
# GET IMAGES
# ============================================================

images = [
    p for p in SOURCE_IMAGES.iterdir()
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
]

if not images:
    print("[ERROR] No images found.")
    exit()

print("=" * 65)
print("TRAFFIC SIGN DATASET SPLIT")
print("=" * 65)

print(f"\nTotal images found: {len(images)}")


# ============================================================
# SHUFFLE
# ============================================================

random.seed(SEED)
random.shuffle(images)


# ============================================================
# CALCULATE SPLIT
# ============================================================

total = len(images)

train_count = int(total * TRAIN_RATIO)
val_count = int(total * VAL_RATIO)
test_count = total - train_count - val_count

train_images = images[:train_count]
val_images = images[train_count:train_count + val_count]
test_images = images[train_count + val_count:]


# ============================================================
# COPY IMAGE + LABEL
# ============================================================

def copy_split(image_list, split):

    copied = 0
    missing_label = 0

    for image_path in image_list:

        label_path = SOURCE_LABELS / f"{image_path.stem}.txt"

        if not label_path.exists():
            print(
                f"[WARNING] Missing label: "
                f"{image_path.name}"
            )
            missing_label += 1
            continue

        destination_image = (
            SOURCE_IMAGES / split / image_path.name
        )

        destination_label = (
            SOURCE_LABELS / split / label_path.name
        )

        shutil.copy2(
            image_path,
            destination_image
        )

        shutil.copy2(
            label_path,
            destination_label
        )

        copied += 1

    print(
        f"{split.upper():5} : "
        f"{copied} images, "
        f"{copied} labels"
    )

    if missing_label:
        print(
            f"       Missing labels: "
            f"{missing_label}"
        )


# ============================================================
# PERFORM SPLIT
# ============================================================

print("\nCreating splits...\n")

copy_split(train_images, "train")
copy_split(val_images, "val")
copy_split(test_images, "test")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("SPLIT COMPLETE")
print("=" * 65)

print(f"""
Train      : {train_count}
Validation : {val_count}
Test       : {test_count}
Total      : {train_count + val_count + test_count}

Ratios:
Train      : 70%
Validation : 20%
Test       : 10%

Random seed: {SEED}
""")

print("Dataset structure:")
print("""
datasets/
└── traffic_sign/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    │
    └── labels/
        ├── train/
        ├── val/
        └── test/
""")

print("=" * 65) 