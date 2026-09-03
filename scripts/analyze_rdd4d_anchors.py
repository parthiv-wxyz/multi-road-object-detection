import json
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "road_damage"
JSON_PATH = DATASET / "annotations" / "voc07_train.json"

CLASS_NAMES = {
    0: "Alligator",
    1: "Block",
    2: "Longitudinal",
    3: "Transversal",
    4: "Pot Hole",
}

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

images = {
    image["id"]: image
    for image in data["images"]
}

boxes = []
class_boxes = defaultdict(list)

for ann in data["annotations"]:

    x, y, w, h = ann["bbox"]

    if w <= 0 or h <= 0:
        continue

    image = images[ann["image_id"]]

    img_w = image["width"]
    img_h = image["height"]

    # Scale boxes as they would appear at 640 x 640
    scaled_w = w / img_w * 640
    scaled_h = h / img_h * 640

    boxes.append([scaled_w, scaled_h])
    class_boxes[ann["category_id"]].append(
        [scaled_w, scaled_h]
    )

boxes = np.array(boxes)

areas = boxes[:, 0] * boxes[:, 1]

print("=" * 70)
print("RDD4D BOUNDING BOX ANALYSIS")
print("=" * 70)

print(f"\nTotal training boxes: {len(boxes)}")

print("\nOverall box dimensions at 640x640:")
print(f"Minimum width : {boxes[:, 0].min():.2f}")
print(f"Maximum width : {boxes[:, 0].max():.2f}")
print(f"Mean width    : {boxes[:, 0].mean():.2f}")
print(f"Median width  : {np.median(boxes[:, 0]):.2f}")

print()

print(f"Minimum height: {boxes[:, 1].min():.2f}")
print(f"Maximum height: {boxes[:, 1].max():.2f}")
print(f"Mean height   : {boxes[:, 1].mean():.2f}")
print(f"Median height : {np.median(boxes[:, 1]):.2f}")

print("\nArea statistics:")
print(f"Minimum area  : {areas.min():.2f}")
print(f"Maximum area  : {areas.max():.2f}")
print(f"Mean area     : {areas.mean():.2f}")
print(f"Median area   : {np.median(areas):.2f}")

print("\n" + "-" * 70)
print("BOX SIZE DISTRIBUTION")
print("-" * 70)

thresholds = [
    (16, "Very small (<16 px)"),
    (32, "Small (<32 px)"),
    (64, "Medium-small (<64 px)"),
    (128, "Medium (<128 px)"),
    (256, "Large (<256 px)"),
]

max_side = np.maximum(boxes[:, 0], boxes[:, 1])

previous = 0

for threshold, label in thresholds:

    count = np.sum(
        (max_side >= previous)
        & (max_side < threshold)
    )

    percentage = count / len(boxes) * 100

    print(
        f"{label:25}: "
        f"{count:5} ({percentage:6.2f}%)"
    )

    previous = threshold

count = np.sum(max_side >= 256)

print(
    f"{'Very large (>=256 px)':25}: "
    f"{count:5} ({count / len(boxes) * 100:6.2f}%)"
)

print("\n" + "-" * 70)
print("CLASS-WISE BOX STATISTICS")
print("-" * 70)

for class_id, name in CLASS_NAMES.items():

    cls_boxes = np.array(class_boxes[class_id])

    if len(cls_boxes) == 0:
        continue

    cls_areas = cls_boxes[:, 0] * cls_boxes[:, 1]

    print(f"\n{name}")
    print(f"  Boxes       : {len(cls_boxes)}")
    print(
        f"  Median W/H  : "
        f"{np.median(cls_boxes[:, 0]):.1f} × "
        f"{np.median(cls_boxes[:, 1]):.1f}"
    )
    print(
        f"  Mean W/H    : "
        f"{cls_boxes[:, 0].mean():.1f} × "
        f"{cls_boxes[:, 1].mean():.1f}"
    )
    print(
        f"  Median area : "
        f"{np.median(cls_areas):.1f}"
    )

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)