"""
visualize_dataset.py

Visualize random YOLO annotations.

Usage:
python scripts/visualize_dataset.py --dataset datasets/idd --samples 20
"""

import argparse
import random
from pathlib import Path

import cv2
import yaml

# ----------------------------------------------------
# Arguments
# ----------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    default="datasets/idd",
    type=str,
)

parser.add_argument(
    "--samples",
    default=20,
    type=int,
)

args = parser.parse_args()

DATASET = Path(args.dataset)
NUM_SAMPLES = args.samples

# ----------------------------------------------------
# Load classes
# ----------------------------------------------------

with open(DATASET / "data.yaml", "r") as f:
    cfg = yaml.safe_load(f)

CLASS_NAMES = cfg["names"]

# ----------------------------------------------------
# Output folder
# ----------------------------------------------------

OUTPUT = Path("results/visualizations")
OUTPUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------
# Colors
# ----------------------------------------------------

random.seed(42)

COLORS = {}

for i in range(len(CLASS_NAMES)):
    COLORS[i] = (
        random.randint(50,255),
        random.randint(50,255),
        random.randint(50,255)
    )

# ----------------------------------------------------
# Collect Images
# ----------------------------------------------------

images = []

for split in ["train","val","test"]:

    img_dir = DATASET / split / "images"

    for ext in ["*.jpg","*.jpeg","*.png","*.bmp"]:

        images.extend(img_dir.glob(ext))
        images.extend(img_dir.glob(ext.upper()))

random.shuffle(images)

images = images[:NUM_SAMPLES]

print(f"Visualizing {len(images)} images...")

# ----------------------------------------------------
# Draw Boxes
# ----------------------------------------------------

saved = 0

for image_path in images:

    split = image_path.parent.parent.name

    label_path = (
        DATASET /
        split /
        "labels" /
        f"{image_path.stem}.txt"
    )

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    H,W = image.shape[:2]

    if label_path.exists():

        with open(label_path) as f:

            for line in f:

                line=line.strip()

                if not line:
                    continue

                cls,x,y,w,h = map(float,line.split())

                cls=int(cls)

                x1=int((x-w/2)*W)
                y1=int((y-h/2)*H)

                x2=int((x+w/2)*W)
                y2=int((y+h/2)*H)

                color = COLORS[cls]

                cv2.rectangle(
                    image,
                    (x1,y1),
                    (x2,y2),
                    color,
                    2
                )

                cv2.putText(
                    image,
                    CLASS_NAMES[cls],
                    (x1,max(25,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

    out_file = OUTPUT / image_path.name

    cv2.imwrite(str(out_file),image)

    saved += 1

print(f"\nSaved {saved} visualization images")

print(f"\nOutput Folder:\n{OUTPUT}")