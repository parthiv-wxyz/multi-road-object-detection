"""
generate_anchors.py

Generate optimized YOLO anchors using K-Means clustering.

Designed for:
datasets/<dataset>/
    images/
        train/
        val/
        test/
    labels/
        train/
        val/
        test/

Example:
python scripts/generate_anchors.py --dataset ../datasets/traffic_sign --img-size 640 --anchors 12
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans


# -------------------------------------------------------
# Arguments
# -------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    default="../datasets/traffic_sign",
    type=str,
    help="Dataset folder",
)

parser.add_argument(
    "--img-size",
    default=640,
    type=int,
    help="Training image size",
)

parser.add_argument(
    "--anchors",
    default=12,
    type=int,
    help="Number of anchors",
)

args = parser.parse_args()

DATASET = Path(args.dataset)
IMG_SIZE = args.img_size
NUM_ANCHORS = args.anchors


# -------------------------------------------------------
# Check dataset
# -------------------------------------------------------

label_dir = DATASET / "labels" / "train"

if not label_dir.exists():
    raise FileNotFoundError(
        f"Training label directory not found:\n{label_dir.resolve()}"
    )

print("\n========================================")
print("Traffic Sign Anchor Generation")
print("========================================")

print(f"Dataset : {DATASET.resolve()}")
print(f"Labels  : {label_dir.resolve()}")
print(f"Image size : {IMG_SIZE}")
print(f"Anchors : {NUM_ANCHORS}")


# -------------------------------------------------------
# Read TRAINING bounding boxes only
# -------------------------------------------------------

boxes = []

label_files = list(label_dir.glob("*.txt"))

print(f"\nTraining label files found: {len(label_files)}")

for label_file in label_files:

    with open(label_file, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            values = line.split()

            if len(values) != 5:
                continue

            cls, x, y, w, h = map(float, values)

            # YOLO normalized width/height → pixels
            box_w = w * IMG_SIZE
            box_h = h * IMG_SIZE

            if box_w > 0 and box_h > 0:
                boxes.append([box_w, box_h])


boxes = np.array(boxes, dtype=np.float32)

if len(boxes) < NUM_ANCHORS:
    raise ValueError(
        f"Only {len(boxes)} bounding boxes found. "
        f"Cannot generate {NUM_ANCHORS} anchors."
    )


print(f"Training bounding boxes: {len(boxes)}")


# -------------------------------------------------------
# KMeans
# -------------------------------------------------------

print("\nRunning K-Means...")

kmeans = KMeans(
    n_clusters=NUM_ANCHORS,
    random_state=42,
    n_init=20,
)

kmeans.fit(boxes)

anchors = kmeans.cluster_centers_


# -------------------------------------------------------
# Sort anchors by area
# -------------------------------------------------------

areas = anchors[:, 0] * anchors[:, 1]

order = np.argsort(areas)

anchors = anchors[order]


# -------------------------------------------------------
# Print anchors
# -------------------------------------------------------

print("\n========================================")
print("Generated Traffic-Sign Anchors")
print("========================================\n")

anchor_list = []

for i, (w, h) in enumerate(anchors, start=1):

    w = round(float(w))
    h = round(float(h))

    anchor_list.append((w, h))

    area = w * h

    print(
        f"{i:2d}. ({w:4d}, {h:4d})"
        f"   area = {area:7d}"
    )


# -------------------------------------------------------
# Save
# -------------------------------------------------------

output_dir = Path("results/anchors")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / f"{DATASET.name}_anchors_{NUM_ANCHORS}.txt"

with open(output_file, "w") as f:

    f.write("Traffic Sign Optimized Anchors\n")
    f.write("================================\n\n")

    f.write(f"Dataset: {DATASET.resolve()}\n")
    f.write(f"Image size: {IMG_SIZE}\n")
    f.write(f"Training boxes: {len(boxes)}\n")
    f.write(f"Number of anchors: {NUM_ANCHORS}\n\n")

    for w, h in anchor_list:
        f.write(f"{w},{h}\n")


# -------------------------------------------------------
# YOLO YAML format
# -------------------------------------------------------

print(f"\nSaved to: {output_file}")

print("\nYOLO YAML anchor format:\n")

for i in range(0, len(anchor_list), 3):

    row = anchor_list[i:i + 3]

    print(
        "  - [" +
        ", ".join(
            f"{w}, {h}" for w, h in row
        ) +
        "]"
    )


print("\n========================================")
print("Anchor generation completed")
print("========================================")