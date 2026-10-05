import os
import csv

ROOT = r"C:\Users\ppart\Desktop\YOLO\datasets\Indian-Traffic Sign-Dataset"
IMAGE_DIR = os.path.join(ROOT, "Images")
CSV_FILE = os.path.join(ROOT, "traffic_sign.csv")

# Read CSV mapping
mapping = {}

with open(CSV_FILE, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)

    for row in reader:
        class_id = int(row["ClassId"])
        name = row["Name"].strip()
        mapping[class_id] = name

print("\nCLASS MAPPING")
print("=" * 80)

total = 0

for class_id in sorted(mapping):
    folder = os.path.join(IMAGE_DIR, str(class_id))

    if os.path.isdir(folder):
        count = len([
            f for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
    else:
        count = 0

    total += count

    print(f"{class_id:2d} | {count:4d} images | {mapping[class_id]}")

print("=" * 80)
print(f"Total images: {total}")
print(f"Total class IDs: {len(mapping)}")
print(f"Unique names: {len(set(mapping.values()))}")