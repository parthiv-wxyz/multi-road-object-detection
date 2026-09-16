from pathlib import Path
import cv2
import random

DATASET = Path(r"E:\Parthiv\multi-road-object-detection\datasets\rdd")

CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Other Corruption",
    "Pothole"
]

SPLIT = "train"
NUM_SAMPLES = 20

images_dir = DATASET / SPLIT / "images"
labels_dir = DATASET / SPLIT / "labels"

image_files = []

for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
    image_files.extend(images_dir.glob(ext))

random.shuffle(image_files)
selected_images = image_files[:NUM_SAMPLES]

for image_path in selected_images:

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    h, w = image.shape[:2]

    label_path = labels_dir / f"{image_path.stem}.txt"

    if label_path.exists():

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            cls, xc, yc, bw, bh = map(float, parts)

            cls = int(cls)

            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                image,
                CLASSES[cls],
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow("RDD2022 Dataset Inspection", image)

    print(f"\nImage: {image_path.name}")
    print(f"Labels: {len(lines) if label_path.exists() else 0}")

    key = cv2.waitKey(0)

    if key == ord("q"):
        break

cv2.destroyAllWindows()