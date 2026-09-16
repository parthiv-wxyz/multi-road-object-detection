import random
import shutil
from pathlib import Path

DATASET = Path("datasets/rdd")

SAMPLES_PER_SPLIT = 50

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def process_split(split):
    images_dir = DATASET / split / "images"
    labels_dir = DATASET / split / "labels"

    output_dir = Path("runs") / "empty_label_audit" / split
    output_dir.mkdir(parents=True, exist_ok=True)

    empty_images = []

    for image_path in images_dir.iterdir():
        if image_path.suffix not in IMAGE_EXTENSIONS:
            continue

        label_path = labels_dir / f"{image_path.stem}.txt"

        if label_path.exists() and label_path.stat().st_size == 0:
            empty_images.append(image_path)

    print(f"\n{split.upper()}")
    print(f"Empty-label images found: {len(empty_images)}")

    if not empty_images:
        return

    samples = random.sample(
        empty_images,
        min(SAMPLES_PER_SPLIT, len(empty_images))
    )

    for image_path in samples:
        shutil.copy2(
            image_path,
            output_dir / image_path.name
        )

    print(f"Copied {len(samples)} samples to:")
    print(output_dir)


def main():
    print("=" * 60)
    print("RDD2022 EMPTY LABEL VISUAL AUDIT")
    print("=" * 60)

    for split in ["train", "val", "test"]:
        process_split(split)

    print("\n" + "=" * 60)
    print("AUDIT SAMPLE CREATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()