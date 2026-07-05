"""
dataset_statistics.py

Generates statistics for a YOLO dataset.

Usage:
python scripts/dataset_statistics.py --dataset datasets/idd
"""

import argparse
from pathlib import Path
from collections import Counter
import yaml
from PIL import Image

# ---------------------------------------------------
# Arguments
# ---------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="datasets/idd",
    help="Dataset directory",
)

args = parser.parse_args()

DATASET = Path(args.dataset)

# ---------------------------------------------------
# Load data.yaml
# ---------------------------------------------------

yaml_file = DATASET / "data.yaml"

with open(yaml_file, "r") as f:
    cfg = yaml.safe_load(f)

NUM_CLASSES = cfg["nc"]
CLASS_NAMES = cfg["names"]

# ---------------------------------------------------
# Statistics
# ---------------------------------------------------

total_images = 0
total_objects = 0

class_counter = Counter()

image_sizes = Counter()

objects_per_image = []

small = 0
medium = 0
large = 0

# ---------------------------------------------------
# Process
# ---------------------------------------------------

for split in ["train", "val", "test"]:

    image_dir = DATASET / split / "images"
    label_dir = DATASET / split / "labels"

    images = []

    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        images.extend(image_dir.glob(ext))
        images.extend(image_dir.glob(ext.upper()))

    total_images += len(images)

    for image_path in images:

        img = Image.open(image_path)

        width, height = img.size

        image_sizes[(width, height)] += 1

        label_file = label_dir / f"{image_path.stem}.txt"

        if not label_file.exists():
            objects_per_image.append(0)
            continue

        with open(label_file) as f:

            lines = [line.strip() for line in f.readlines() if line.strip()]

        objects_per_image.append(len(lines))

        for line in lines:

            cls, x, y, w, h = map(float, line.split())

            cls = int(cls)

            class_counter[cls] += 1
            total_objects += 1

            area = w * h

            if area < 0.01:
                small += 1
            elif area < 0.09:
                medium += 1
            else:
                large += 1

# ---------------------------------------------------
# Print Report
# ---------------------------------------------------

print("=" * 60)
print("DATASET STATISTICS")
print("=" * 60)

print(f"Dataset           : {DATASET}")
print(f"Images            : {total_images}")
print(f"Objects           : {total_objects}")

print()

print("Objects per Class")

print("-" * 60)

for i in range(NUM_CLASSES):
    print(f"{i:2d} {CLASS_NAMES[i]:20s} {class_counter[i]}")

print()

print(f"Average Objects/Image : {sum(objects_per_image)/len(objects_per_image):.2f}")

print()

print("Bounding Box Size")

print(f"Small  : {small}")
print(f"Medium : {medium}")
print(f"Large  : {large}")

print()

print("Image Resolutions")

for size, count in image_sizes.most_common(10):
    print(f"{size[0]} x {size[1]} : {count}")

print("=" * 60)

# ---------------------------------------------------
# Save Report
# ---------------------------------------------------

report_dir = Path("results/reports")
report_dir.mkdir(parents=True, exist_ok=True)

report_file = report_dir / f"{DATASET.name}_statistics.txt"

with open(report_file, "w") as f:

    f.write("=" * 60 + "\n")
    f.write("DATASET STATISTICS\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Dataset : {DATASET}\n")
    f.write(f"Images : {total_images}\n")
    f.write(f"Objects : {total_objects}\n\n")

    f.write("Objects Per Class\n")
    f.write("-" * 60 + "\n")

    for i in range(NUM_CLASSES):
        f.write(f"{i:2d} {CLASS_NAMES[i]:20s} {class_counter[i]}\n")

    f.write("\n")
    f.write(f"Average Objects/Image : {sum(objects_per_image)/len(objects_per_image):.2f}\n\n")

    f.write("Bounding Box Size\n")
    f.write(f"Small : {small}\n")
    f.write(f"Medium : {medium}\n")
    f.write(f"Large : {large}\n")

print(f"\nReport saved to {report_file}")