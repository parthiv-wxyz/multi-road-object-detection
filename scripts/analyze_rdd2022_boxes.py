from pathlib import Path
import cv2
import numpy as np

DATASET = Path(r"E:\Parthiv\multi-road-object-detection\datasets\rdd")
SPLIT = "train"

CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Other Corruption",
    "Pothole"
]

images_dir = DATASET / SPLIT / "images"
labels_dir = DATASET / SPLIT / "labels"

extensions = {".jpg", ".jpeg", ".png"}

image_files = [
    p for p in images_dir.iterdir()
    if p.is_file() and p.suffix.lower() in extensions
]

all_boxes = []
class_boxes = {i: [] for i in range(len(CLASSES))}

for image_path in image_files:

    label_path = labels_dir / f"{image_path.stem}.txt"

    if not label_path.exists():
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    h, w = image.shape[:2]

    for line in label_path.read_text().splitlines():

        parts = line.strip().split()

        if len(parts) != 5:
            continue

        cls, xc, yc, bw, bh = map(float, parts)
        cls = int(cls)

        # Convert normalized dimensions to equivalent
        # dimensions at 640 x 640
        box_w = bw * 640
        box_h = bh * 640

        area = box_w * box_h

        all_boxes.append((box_w, box_h, area))
        class_boxes[cls].append((box_w, box_h, area))


def statistics(values):

    values = np.array(values)

    return {
        "min": np.min(values),
        "max": np.max(values),
        "mean": np.mean(values),
        "median": np.median(values)
    }


print("=" * 70)
print("RDD2022 BOUNDING BOX ANALYSIS")
print("=" * 70)

widths = [x[0] for x in all_boxes]
heights = [x[1] for x in all_boxes]
areas = [x[2] for x in all_boxes]

print(f"\nTotal training boxes: {len(all_boxes)}")

print("\nOverall dimensions at 640 × 640:")

for name, values in [
    ("Width", widths),
    ("Height", heights),
    ("Area", areas)
]:

    s = statistics(values)

    print(f"\n{name}")
    print(f"  Minimum : {s['min']:.2f}")
    print(f"  Maximum : {s['max']:.2f}")
    print(f"  Mean    : {s['mean']:.2f}")
    print(f"  Median  : {s['median']:.2f}")


# Size distribution based on maximum dimension
print("\n" + "-" * 70)
print("BOX SIZE DISTRIBUTION")
print("-" * 70)

sizes = np.maximum(widths, heights)

categories = [
    ("Very small (<16 px)", sizes < 16),
    ("Small (<32 px)", (sizes >= 16) & (sizes < 32)),
    ("Medium-small (<64 px)", (sizes >= 32) & (sizes < 64)),
    ("Medium (<128 px)", (sizes >= 64) & (sizes < 128)),
    ("Large (<256 px)", (sizes >= 128) & (sizes < 256)),
    ("Very large (>=256 px)", sizes >= 256)
]

total = len(sizes)

for name, mask in categories:

    count = np.sum(mask)

    print(
        f"{name:<30}: "
        f"{count:>6} "
        f"({count / total * 100:>6.2f}%)"
    )


print("\n" + "-" * 70)
print("CLASS-WISE BOX STATISTICS")
print("-" * 70)

for cls, name in enumerate(CLASSES):

    boxes = class_boxes[cls]

    if not boxes:
        continue

    widths = np.array([x[0] for x in boxes])
    heights = np.array([x[1] for x in boxes])
    areas = np.array([x[2] for x in boxes])

    print(f"\n{name}")
    print(f"  Boxes       : {len(boxes)}")
    print(
        f"  Median W/H  : "
        f"{np.median(widths):.1f} × "
        f"{np.median(heights):.1f}"
    )
    print(
        f"  Mean W/H    : "
        f"{np.mean(widths):.1f} × "
        f"{np.mean(heights):.1f}"
    )
    print(
        f"  Median area : "
        f"{np.median(areas):.1f}"
    )

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)