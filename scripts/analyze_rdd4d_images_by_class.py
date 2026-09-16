import os
from pathlib import Path
from collections import defaultdict

# Dataset location
DATASET = Path(r"E:\Parthiv\multi-road-object-detection\datasets\road_damage")

# Class names
CLASSES = {
    0: "Alligator",
    1: "Block",
    2: "Longitudinal",
    3: "Transversal",
    4: "Pot Hole"
}

SPLITS = ["train", "val", "test"]

print("=" * 70)
print("RDD4D IMAGE-WISE CLASS ANALYSIS")
print("=" * 70)

# Store unique images containing each class
class_images = defaultdict(set)

# Store multi-class information
image_classes = {}

for split in SPLITS:

    label_dir = DATASET / "labels" / split

    print(f"\nScanning {split.upper()}...")

    if not label_dir.exists():
        print(f"WARNING: Directory not found: {label_dir}")
        continue

    for label_file in label_dir.glob("*.txt"):

        classes_in_image = set()

        with open(label_file, "r") as f:
            for line in f:

                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                if len(parts) != 5:
                    continue

                class_id = int(parts[0])

                classes_in_image.add(class_id)
                class_images[class_id].add(
                    f"{split}/{label_file.stem}"
                )

        image_classes[f"{split}/{label_file.stem}"] = classes_in_image


# Print results
print("\n" + "=" * 70)
print("UNIQUE IMAGES CONTAINING EACH CLASS")
print("=" * 70)

for class_id, class_name in CLASSES.items():

    total = len(class_images[class_id])

    print(f"\n{class_id} -> {class_name}")
    print(f"Unique images: {total}")

    for split in SPLITS:

        count = sum(
            1
            for image_id in class_images[class_id]
            if image_id.startswith(f"{split}/")
        )

        print(f"  {split.upper():<6}: {count}")


# Multi-class images
print("\n" + "=" * 70)
print("IMAGE CLASS COMBINATIONS")
print("=" * 70)

combinations = defaultdict(int)

for image_id, classes in image_classes.items():

    if classes:
        class_names = tuple(
            sorted(CLASSES[class_id] for class_id in classes)
        )

        combinations[class_names] += 1


for combination, count in sorted(
    combinations.items(),
    key=lambda x: x[1],
    reverse=True
):

    print(f"{count:4} images -> {', '.join(combination)}")


print("\n" + "=" * 70)
print("RARE CLASS IMAGES")
print("=" * 70)

for class_id in [0, 1, 4]:

    class_name = CLASSES[class_id]

    print(f"\n{class_name} images:")

    images = sorted(class_images[class_id])

    for image in images:
        print(f"  {image}")


print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)