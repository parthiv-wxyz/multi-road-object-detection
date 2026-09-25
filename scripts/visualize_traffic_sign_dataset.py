from pathlib import Path
import random
import cv2

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "datasets" / "traffic_sign"

SPLIT = "train"
NUM_IMAGES = 12

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

CLASS_NAMES = {
    0: "regulatory",
    1: "mandatory",
    2: "informatory",
    3: "general",
    4: "warning",
}


# ============================================================
# FIND IMAGES
# ============================================================

images_dir = DATASET_ROOT / "images" / SPLIT
labels_dir = DATASET_ROOT / "labels" / SPLIT

images = [
    p for p in images_dir.iterdir()
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
]

if not images:
    print("[ERROR] No images found.")
    exit()

random.seed(42)
selected = random.sample(
    images,
    min(NUM_IMAGES, len(images))
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

output_dir = DATASET_ROOT / "visualization"
output_dir.mkdir(exist_ok=True)


# ============================================================
# DRAW BOXES
# ============================================================

for image_path in selected:

    label_path = labels_dir / f"{image_path.stem}.txt"

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"[ERROR] Cannot read {image_path.name}")
        continue

    height, width = image.shape[:2]

    if label_path.exists():

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as f:

            lines = [
                line.strip()
                for line in f
                if line.strip()
            ]

        for line in lines:

            parts = line.split()

            if len(parts) != 5:
                continue

            class_id = int(float(parts[0]))

            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])

            # YOLO normalized coordinates
            x_center *= width
            y_center *= height
            box_width *= width
            box_height *= height

            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)

            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            # Keep coordinates inside image
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width - 1, x2)
            y2 = min(height - 1, y2)

            # Draw bounding box
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            label = CLASS_NAMES.get(
                class_id,
                f"class_{class_id}"
            )

            cv2.putText(
                image,
                label,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

    output_path = output_dir / image_path.name

    cv2.imwrite(
        str(output_path),
        image
    )

    print(f"[OK] {output_path}")


print("\n" + "=" * 60)
print("VISUALIZATION COMPLETE")
print("=" * 60)

print(f"\nImages saved to:")
print(output_dir)