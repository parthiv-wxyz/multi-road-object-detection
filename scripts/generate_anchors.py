"""
generate_anchors.py

Generate optimized YOLO anchors using K-Means clustering.

Usage:
python scripts/generate_anchors.py --dataset datasets/idd --img-size 640 --anchors 9
"""

import argparse
from pathlib import Path

import numpy as np
import yaml
from sklearn.cluster import KMeans

# -------------------------------------------------------
# Arguments
# -------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    default="datasets/idd",
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
    default=9,
    type=int,
    help="Number of anchors",
)

args = parser.parse_args()

DATASET = Path(args.dataset)
IMG_SIZE = args.img_size
NUM_ANCHORS = args.anchors

# -------------------------------------------------------
# Load dataset
# -------------------------------------------------------

yaml_file = DATASET / "data.yaml"

with open(yaml_file, "r") as f:
    cfg = yaml.safe_load(f)

# -------------------------------------------------------
# Read all bounding boxes
# -------------------------------------------------------

boxes = []

for split in ["train", "val", "test"]:

    label_dir = DATASET / split / "labels"

    if not label_dir.exists():
        continue

    for label_file in label_dir.glob("*.txt"):

        with open(label_file) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                cls, x, y, w, h = map(float, line.split())

                boxes.append([
                    w * IMG_SIZE,
                    h * IMG_SIZE,
                ])

boxes = np.array(boxes)

print(f"\nLoaded {len(boxes)} bounding boxes")

# -------------------------------------------------------
# KMeans
# -------------------------------------------------------

kmeans = KMeans(
    n_clusters=NUM_ANCHORS,
    random_state=42,
    n_init=20,
)

kmeans.fit(boxes)

anchors = kmeans.cluster_centers_

# -------------------------------------------------------
# Sort by area
# -------------------------------------------------------

areas = anchors[:, 0] * anchors[:, 1]

order = np.argsort(areas)

anchors = anchors[order]

# -------------------------------------------------------
# Print
# -------------------------------------------------------

print("\nGenerated Anchors\n")

anchor_list = []

for i, (w, h) in enumerate(anchors, start=1):

    w = round(float(w))
    h = round(float(h))

    anchor_list.append((w, h))

    print(f"{i:2d}. ({w:3d}, {h:3d})")

# -------------------------------------------------------
# Save
# -------------------------------------------------------

output_dir = Path("results/anchors")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / f"{DATASET.name}_anchors.txt"

with open(output_file, "w") as f:

    f.write("Optimized Anchors\n\n")

    for w, h in anchor_list:
        f.write(f"{w},{h}\n")

print(f"\nSaved to {output_file}")

print("\nYOLO format:\n")

for i in range(0, len(anchor_list), 3):

    row = anchor_list[i:i + 3]

    print(" ".join(f"{w},{h}" for w, h in row))