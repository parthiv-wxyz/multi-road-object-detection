from pathlib import Path
import cv2
import random

DATASET = Path(r"E:\Parthiv\multi-road-object-detection\datasets\rdd")

CLASSES = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Other Corruption",
    4: "Pothole"
}

SPLIT = "train"

images_dir = DATASET / SPLIT / "images"
labels_dir = DATASET / SPLIT / "labels"

# Number of images to inspect per category
SAMPLES_PER_CLASS = 5
EMPTY_SAMPLES = 10


def get_image_files():
    files = []

    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        files.extend(images_dir.glob(ext))

    return files


def read_labels(image_path):

    label_path = labels_dir / f"{image_path.stem}.txt"

    if not label_path.exists():
        return []

    lines = label_path.read_text().strip().splitlines()

    labels = []

    for line in lines:

        parts = line.strip().split()

        if len(parts) != 5:
            continue

        cls, xc, yc, bw, bh = map(float, parts)

        labels.append(
            (int(cls), xc, yc, bw, bh)
        )

    return labels


def draw_labels(image, labels):

    h, w = image.shape[:2]

    for cls, xc, yc, bw, bh in labels:

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
            0.7,
            (0, 255, 0),
            2
        )

    return image


image_files = get_image_files()

print("=" * 70)
print("RDD2022 TARGETED DATASET INSPECTION")
print("=" * 70)

# --------------------------------------------------
# EMPTY LABEL IMAGES
# --------------------------------------------------

empty_images = []

for image_path in image_files:

    labels = read_labels(image_path)

    if len(labels) == 0:
        empty_images.append(image_path)

random.shuffle(empty_images)

print(f"\nEmpty-label images found: {len(empty_images)}")

samples = empty_images[:EMPTY_SAMPLES]

for image_path in samples:

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    print(f"\n[EMPTY] {image_path.name}")

    cv2.imshow("RDD2022 Inspection", image)

    key = cv2.waitKey(0)

    if key == ord("q"):
        cv2.destroyAllWindows()
        exit()

# --------------------------------------------------
# CLASS-SPECIFIC IMAGES
# --------------------------------------------------

for target_class, class_name in CLASSES.items():

    class_images = []

    for image_path in image_files:

        labels = read_labels(image_path)

        if any(label[0] == target_class for label in labels):
            class_images.append(image_path)

    random.shuffle(class_images)

    print("\n" + "=" * 70)
    print(f"CLASS: {class_name}")
    print(f"Images containing this class: {len(class_images)}")
    print("=" * 70)

    samples = class_images[:SAMPLES_PER_CLASS]

    for image_path in samples:

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        labels = read_labels(image_path)

        image = draw_labels(image, labels)

        print(f"\n[{class_name}] {image_path.name}")
        print(f"Objects in image: {len(labels)}")

        cv2.imshow(
            "RDD2022 Inspection",
            image
        )

        key = cv2.waitKey(0)

        if key == ord("q"):
            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()

print("\nInspection complete.")