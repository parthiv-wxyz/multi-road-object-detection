import os
import shutil
import random

SOURCE = r"C:\Users\ppart\Desktop\YOLO\datasets\Indian-Traffic Sign-Dataset\Images"

DEST = r"C:\Users\ppart\Desktop\YOLO\datasets\Indian-Traffic-Sign-Classification"

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

SEED = 42

random.seed(SEED)

# Create destination directories
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(DEST, split), exist_ok=True)

total_images = 0
total_classes = 0

print("=" * 70)
print("CREATING 80/10/10 CLASSIFICATION DATASET")
print("=" * 70)

for class_id in sorted(os.listdir(SOURCE), key=lambda x: int(x)):
    
    class_source = os.path.join(SOURCE, class_id)

    if not os.path.isdir(class_source):
        continue

    # Get PNG/JPG/JPEG images
    images = [
        f for f in os.listdir(class_source)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    # Skip empty classes
    if len(images) == 0:
        print(f"Class {class_id}: SKIPPED (0 images)")
        continue

    random.shuffle(images)

    n = len(images)

    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    # Create class directories
    for split in ["train", "val", "test"]:
        os.makedirs(
            os.path.join(DEST, split, class_id),
            exist_ok=True
        )

    # Copy files
    for image in train_images:
        shutil.copy2(
            os.path.join(class_source, image),
            os.path.join(DEST, "train", class_id, image)
        )

    for image in val_images:
        shutil.copy2(
            os.path.join(class_source, image),
            os.path.join(DEST, "val", class_id, image)
        )

    for image in test_images:
        shutil.copy2(
            os.path.join(class_source, image),
            os.path.join(DEST, "test", class_id, image)
        )

    total_images += n
    total_classes += 1

    print(
        f"Class {class_id:>2}: "
        f"{n:>4} total | "
        f"Train {len(train_images):>4} | "
        f"Val {len(val_images):>4} | "
        f"Test {len(test_images):>4}"
    )

print("=" * 70)
print(f"Classes: {total_classes}")
print(f"Images : {total_images}")
print("Split  : 80% train / 10% val / 10% test")
print(f"Output : {DEST}")
print("=" * 70)