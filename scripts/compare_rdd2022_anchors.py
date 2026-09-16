import numpy as np
from pathlib import Path


LABEL_DIR = Path("datasets/rdd/train/labels")
IMG_SIZE = 640


# Default anchors from the current enhanced model
CURRENT_ANCHORS = np.array([
    [54, 19], [32, 64], [93, 29],
    [65, 87], [187, 35], [139, 58],
    [52, 203], [108, 152], [325, 68],
    [170, 293], [490, 127], [263, 524]
])


# Newly optimized RDD2022 anchors
OPTIMIZED_ANCHORS = np.array([
    [35, 33], [109, 41], [54, 112],
    [217, 51], [134, 132], [86, 249],
    [368, 96], [216, 193], [123, 542],
    [573, 135], [301, 285], [542, 315]
])


def load_boxes():
    boxes = []

    for label_file in LABEL_DIR.glob("*.txt"):
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                _, _, _, w, h = map(float, parts)

                if w <= 0 or h <= 0:
                    continue

                boxes.append([
                    w * IMG_SIZE,
                    h * IMG_SIZE
                ])

    return np.array(boxes)


def wh_iou(boxes, anchors):

    boxes = boxes[:, None, :]
    anchors = anchors[None, :, :]

    intersection = np.minimum(boxes, anchors).prod(axis=2)

    box_area = boxes.prod(axis=2)
    anchor_area = anchors.prod(axis=2)

    union = box_area + anchor_area - intersection

    return intersection / union


def evaluate(name, boxes, anchors):

    ious = wh_iou(boxes, anchors)

    best_iou = ious.max(axis=1)

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Mean Best IoU   : {best_iou.mean():.4f}")
    print(f"Median Best IoU : {np.median(best_iou):.4f}")

    for threshold in [0.25, 0.50, 0.60, 0.70]:
        count = (best_iou >= threshold).sum()
        percentage = count / len(best_iou) * 100

        print(
            f"IoU >= {threshold:.2f}: "
            f"{count}/{len(best_iou)} "
            f"({percentage:.2f}%)"
        )


def main():

    print("=" * 60)
    print("RDD2022 ANCHOR COMPARISON")
    print("=" * 60)

    boxes = load_boxes()

    print(f"\nTraining boxes: {len(boxes)}")

    evaluate(
        "CURRENT ENHANCED MODEL ANCHORS",
        boxes,
        CURRENT_ANCHORS
    )

    evaluate(
        "RDD2022 OPTIMIZED ANCHORS",
        boxes,
        OPTIMIZED_ANCHORS
    )


if __name__ == "__main__":
    main()