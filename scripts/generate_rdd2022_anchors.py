from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans

DATASET = Path(r"E:\Parthiv\multi-road-object-detection\datasets\rdd")

IMG_SIZE = 640
NUM_ANCHORS = 12

images_dir = DATASET / "train" / "images"
labels_dir = DATASET / "train" / "labels"

boxes = []

print("=" * 70)
print("RDD2022 DATASET-SPECIFIC ANCHOR OPTIMIZATION")
print("=" * 70)

for label_path in labels_dir.glob("*.txt"):

    for line in label_path.read_text().splitlines():

        parts = line.strip().split()

        if len(parts) != 5:
            continue

        _, _, _, bw, bh = map(float, parts)

        if bw <= 0 or bh <= 0:
            continue

        boxes.append([
            bw * IMG_SIZE,
            bh * IMG_SIZE
        ])

boxes = np.array(boxes, dtype=np.float32)

print(f"\nTraining boxes: {len(boxes)}")
print(f"Training size: {IMG_SIZE} × {IMG_SIZE}")
print(f"Anchors requested: {NUM_ANCHORS}")

if len(boxes) < NUM_ANCHORS:
    raise ValueError("Not enough bounding boxes for anchor generation.")


# ------------------------------------------------------------
# IoU calculation using width and height only
# ------------------------------------------------------------

def wh_iou(boxes, anchors):

    boxes = boxes[:, None, :]
    anchors = anchors[None, :, :]

    intersection = np.minimum(boxes, anchors).prod(axis=2)

    box_area = boxes.prod(axis=2)
    anchor_area = anchors.prod(axis=2)

    union = box_area + anchor_area - intersection

    return intersection / (union + 1e-9)


# ------------------------------------------------------------
# K-means clustering
# ------------------------------------------------------------

kmeans = KMeans(
    n_clusters=NUM_ANCHORS,
    random_state=42,
    n_init=20
)

kmeans.fit(boxes)

anchors = kmeans.cluster_centers_

# Sort by area
areas = anchors[:, 0] * anchors[:, 1]
anchors = anchors[np.argsort(areas)]


# ------------------------------------------------------------
# Calculate mean best IoU
# ------------------------------------------------------------

ious = wh_iou(boxes, anchors)

best_ious = ious.max(axis=1)

print("\n" + "-" * 70)
print("OPTIMIZED ANCHORS")
print("-" * 70)

for i, (w, h) in enumerate(anchors, start=1):

    print(
        f"Anchor {i:2}: "
        f"{w:6.1f} × {h:6.1f}"
    )

print(f"\nMean Best IoU: {best_ious.mean():.4f}")

print("\n" + "=" * 70)
print("YOLO YAML FORMAT")
print("=" * 70)

anchors_int = np.round(anchors).astype(int)

for i in range(0, NUM_ANCHORS, 3):

    group = anchors_int[i:i + 3].flatten().tolist()

    level = ["P2", "P3", "P4", "P5"][i // 3]

    print(
        f"  - {group} # {level}"
    )

print("\n" + "=" * 70)
print("OPTIMIZATION COMPLETE")
print("=" * 70)