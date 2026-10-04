import os
import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = Path(
    r"C:\Users\ppart\Desktop\YOLO\datasets\archive"
)

OUTPUT_FILE = DATASET_PATH.parent / "archive_dataset_analysis.txt"


# ============================================================
# FILE TYPES
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff"
}

ANNOTATION_EXTENSIONS = {
    ".txt", ".xml", ".json",
    ".csv", ".yaml", ".yml"
}

ARCHIVE_EXTENSIONS = {
    ".zip", ".rar", ".7z", ".tar", ".gz"
}


# ============================================================
# HELPERS
# ============================================================

def format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"

    if size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.2f} KB"

    if size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.2f} MB"

    return f"{size_bytes / (1024 ** 3):.2f} GB"


def file_hash(path, chunk_size=1024 * 1024):
    sha256 = hashlib.sha256()

    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)

                if not chunk:
                    break

                sha256.update(chunk)

        return sha256.hexdigest()

    except Exception:
        return None


def is_image(path):
    return path.suffix.lower() in IMAGE_EXTENSIONS


def is_annotation(path):
    return path.suffix.lower() in ANNOTATION_EXTENSIONS


def safe_relative(path):
    try:
        return path.relative_to(DATASET_PATH)
    except ValueError:
        return path


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze_dataset():

    if not DATASET_PATH.exists():

        print("ERROR: Dataset folder does not exist:")
        print(DATASET_PATH)

        return

    print("Analyzing dataset...")
    print("Please wait. Large datasets may take some time.")

    report = []

    def add(text=""):
        report.append(text)

    # ========================================================
    # HEADER
    # ========================================================

    add("=" * 80)
    add("ROAD DAMAGE DATASET - COMPLETE ANALYSIS REPORT")
    add("=" * 80)

    add(f"Dataset path:")
    add(str(DATASET_PATH))
    add()

    # ========================================================
    # COLLECT FILES
    # ========================================================

    all_files = []
    all_directories = []

    for root, dirs, files in os.walk(DATASET_PATH):

        root_path = Path(root)

        for d in dirs:
            all_directories.append(root_path / d)

        for filename in files:
            all_files.append(root_path / filename)

    # ========================================================
    # BASIC FILE STATISTICS
    # ========================================================

    total_files = len(all_files)
    total_directories = len(all_directories)

    total_size = 0

    for file in all_files:
        try:
            total_size += file.stat().st_size
        except Exception:
            pass

    image_files = [
        f for f in all_files
        if is_image(f)
    ]

    annotation_files = [
        f for f in all_files
        if is_annotation(f)
    ]

    archive_files = [
        f for f in all_files
        if f.suffix.lower() in ARCHIVE_EXTENSIONS
    ]

    other_files = [
        f for f in all_files
        if not is_image(f)
        and not is_annotation(f)
        and f.suffix.lower() not in ARCHIVE_EXTENSIONS
    ]

    add("=" * 80)
    add("1. BASIC DATASET INFORMATION")
    add("=" * 80)

    add(f"Dataset directory       : {DATASET_PATH}")
    add(f"Total directories       : {total_directories}")
    add(f"Total files             : {total_files}")
    add(f"Total dataset size      : {format_size(total_size)}")
    add(f"Image files             : {len(image_files)}")
    add(f"Annotation/config files : {len(annotation_files)}")
    add(f"Archive files           : {len(archive_files)}")
    add(f"Other files             : {len(other_files)}")
    add()

    # ========================================================
    # EXTENSION DISTRIBUTION
    # ========================================================

    extension_counter = Counter(
        f.suffix.lower() if f.suffix else "[NO EXTENSION]"
        for f in all_files
    )

    add("=" * 80)
    add("2. FILE TYPE DISTRIBUTION")
    add("=" * 80)

    for ext, count in extension_counter.most_common():
        add(f"{ext:20} : {count}")

    add()

    # ========================================================
    # DIRECTORY STRUCTURE
    # ========================================================

    add("=" * 80)
    add("3. COMPLETE DIRECTORY STRUCTURE")
    add("=" * 80)

    try:
        structure = []

        for path in sorted(DATASET_PATH.rglob("*")):

            relative = path.relative_to(DATASET_PATH)

            depth = len(relative.parts) - 1

            if path.is_dir():
                structure.append(
                    "    " * depth +
                    f"[DIR]  {path.name}"
                )

            else:
                structure.append(
                    "    " * depth +
                    f"[FILE] {path.name}"
                )

        for line in structure:
            add(line)

    except Exception as e:
        add(f"Could not generate directory tree: {e}")

    add()

    # ========================================================
    # IMAGES
    # ========================================================

    add("=" * 80)
    add("4. IMAGE ANALYSIS")
    add("=" * 80)

    if not image_files:

        add("No image files detected.")

    else:

        image_extensions = Counter(
            f.suffix.lower()
            for f in image_files
        )

        add(f"Total images: {len(image_files)}")
        add()

        add("Image formats:")

        for ext, count in image_extensions.most_common():
            add(f"  {ext:10} : {count}")

        add()

        widths = []
        heights = []
        image_sizes = []
        corrupted_images = []

        if PIL_AVAILABLE:

            for image_path in image_files:

                try:

                    with Image.open(image_path) as img:

                        width, height = img.size

                        widths.append(width)
                        heights.append(height)

                        image_sizes.append(
                            width * height
                        )

                except Exception as e:

                    corrupted_images.append(
                        (image_path, str(e))
                    )

            if widths:

                add("Image resolution statistics:")
                add(f"  Minimum width  : {min(widths)}")
                add(f"  Maximum width  : {max(widths)}")
                add(f"  Average width  : {sum(widths) / len(widths):.2f}")
                add()

                add(f"  Minimum height : {min(heights)}")
                add(f"  Maximum height : {max(heights)}")
                add(f"  Average height : {sum(heights) / len(heights):.2f}")
                add()

                resolution_counter = Counter()

                for image_path in image_files:

                    try:

                        with Image.open(image_path) as img:

                            resolution_counter[
                                f"{img.width}x{img.height}"
                            ] += 1

                    except Exception:
                        pass

                add("Resolution distribution:")

                for resolution, count in resolution_counter.most_common():

                    add(
                        f"  {resolution:15} : {count}"
                    )

                add()

            add(
                f"Corrupted/unreadable images: "
                f"{len(corrupted_images)}"
            )

            if corrupted_images:

                for path, error in corrupted_images:

                    add(
                        f"  {safe_relative(path)}"
                    )

        else:

            add(
                "Pillow is not installed."
            )

            add(
                "Install Pillow to obtain image dimensions."
            )

    add()

    # ========================================================
    # IMAGE SIZE DISTRIBUTION
    # ========================================================

    add("=" * 80)
    add("5. IMAGE FILE SIZE ANALYSIS")
    add("=" * 80)

    if image_files:

        sizes = []

        for path in image_files:

            try:
                sizes.append(path.stat().st_size)

            except Exception:
                pass

        if sizes:

            add(
                f"Smallest image : "
                f"{format_size(min(sizes))}"
            )

            add(
                f"Largest image  : "
                f"{format_size(max(sizes))}"
            )

            add(
                f"Average image  : "
                f"{format_size(sum(sizes) / len(sizes))}"
            )

    add()

    # ========================================================
    # ANNOTATION ANALYSIS
    # ========================================================

    add("=" * 80)
    add("6. ANNOTATION / LABEL ANALYSIS")
    add("=" * 80)

    if not annotation_files:

        add("No annotation/config files detected.")

    else:

        annotation_extensions = Counter(
            f.suffix.lower()
            for f in annotation_files
        )

        add(
            f"Total annotation/config files: "
            f"{len(annotation_files)}"
        )

        add()

        for ext, count in annotation_extensions.most_common():

            add(
                f"{ext:10} : {count}"
            )

    add()

    # ========================================================
    # YOLO TXT ANALYSIS
    # ========================================================

    txt_files = [
        f for f in annotation_files
        if f.suffix.lower() == ".txt"
    ]

    yolo_class_counter = Counter()

    total_yolo_boxes = 0
    invalid_yolo_lines = 0
    empty_label_files = 0

    yolo_label_files = []

    for label_path in txt_files:

        try:

            with open(
                label_path,
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as f:

                lines = [
                    line.strip()
                    for line in f
                    if line.strip()
                ]

            if not lines:

                empty_label_files += 1
                continue

            valid_lines = 0

            for line in lines:

                parts = line.split()

                # Standard YOLO:
                # class x_center y_center width height

                if len(parts) >= 5:

                    try:

                        class_id = int(float(parts[0]))

                        x = float(parts[1])
                        y = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])

                        if (
                            0 <= x <= 1
                            and 0 <= y <= 1
                            and 0 < w <= 1
                            and 0 < h <= 1
                        ):

                            yolo_class_counter[class_id] += 1

                            total_yolo_boxes += 1

                            valid_lines += 1

                        else:

                            invalid_yolo_lines += 1

                    except Exception:

                        invalid_yolo_lines += 1

                else:

                    invalid_yolo_lines += 1

            yolo_label_files.append(label_path)

        except Exception:

            pass

    if txt_files:

        add("=" * 80)
        add("7. YOLO TXT ANNOTATION ANALYSIS")
        add("=" * 80)

        add(
            f"TXT files checked       : {len(txt_files)}"
        )

        add(
            f"Possible YOLO labels    : "
            f"{len(yolo_label_files)}"
        )

        add(
            f"Total bounding boxes    : "
            f"{total_yolo_boxes}"
        )

        add(
            f"Empty label files       : "
            f"{empty_label_files}"
        )

        add(
            f"Invalid annotation lines : "
            f"{invalid_yolo_lines}"
        )

        add()

        if yolo_class_counter:

            add("YOLO class distribution:")

            total_objects = sum(
                yolo_class_counter.values()
            )

            for class_id, count in sorted(
                yolo_class_counter.items()
            ):

                percentage = (
                    count / total_objects * 100
                )

                add(
                    f"  Class {class_id:<5} : "
                    f"{count:<8} "
                    f"({percentage:.2f}%)"
                )

        add()

    # ========================================================
    # JSON ANALYSIS
    # ========================================================

    json_files = [
        f for f in annotation_files
        if f.suffix.lower() == ".json"
    ]

    add("=" * 80)
    add("8. JSON FILE ANALYSIS")
    add("=" * 80)

    if not json_files:

        add("No JSON files found.")

    else:

        add(
            f"JSON files found: {len(json_files)}"
        )

        for json_path in json_files:

            add()
            add(
                f"JSON: {safe_relative(json_path)}"
            )

            try:

                with open(
                    json_path,
                    "r",
                    encoding="utf-8",
                    errors="ignore"
                ) as f:

                    data = json.load(f)

                add(
                    f"  Root type: "
                    f"{type(data).__name__}"
                )

                if isinstance(data, dict):

                    add(
                        "  Keys:"
                    )

                    for key in data.keys():

                        add(
                            f"    - {key}"
                        )

                    # COCO-style dataset

                    if (
                        "images" in data
                        and "annotations" in data
                    ):

                        add()
                        add(
                            "  COCO-style structure detected."
                        )

                        add(
                            f"  Images      : "
                            f"{len(data['images'])}"
                        )

                        add(
                            f"  Annotations : "
                            f"{len(data['annotations'])}"
                        )

                        if "categories" in data:

                            add(
                                f"  Categories  : "
                                f"{len(data['categories'])}"
                            )

                            add(
                                "  Category list:"
                            )

                            for category in data["categories"]:

                                add(
                                    f"    "
                                    f"{category.get('id')} : "
                                    f"{category.get('name')}"
                                )

                        json_category_counter = Counter()

                        for annotation in data["annotations"]:

                            if "category_id" in annotation:

                                json_category_counter[
                                    annotation["category_id"]
                                ] += 1

                        if json_category_counter:

                            add()
                            add(
                                "  Annotation distribution:"
                            )

                            for cid, count in sorted(
                                json_category_counter.items()
                            ):

                                add(
                                    f"    Class {cid}: "
                                    f"{count}"
                                )

                    # Pascal/VOC-like

                    elif (
                        "annotations" in data
                        and "categories" in data
                    ):

                        add(
                            "  Annotation/category structure detected."
                        )

                elif isinstance(data, list):

                    add(
                        f"  JSON contains "
                        f"{len(data)} top-level entries."
                    )

            except Exception as e:

                add(
                    f"  ERROR reading JSON: {e}"
                )

    add()

    # ========================================================
    # XML ANALYSIS
    # ========================================================

    xml_files = [
        f for f in annotation_files
        if f.suffix.lower() == ".xml"
    ]

    add("=" * 80)
    add("9. XML / PASCAL VOC ANALYSIS")
    add("=" * 80)

    if not xml_files:

        add("No XML files found.")

    else:

        add(
            f"XML files found: {len(xml_files)}"
        )

        try:

            import xml.etree.ElementTree as ET

            voc_class_counter = Counter()
            voc_boxes = 0
            voc_invalid = 0

            for xml_path in xml_files:

                try:

                    tree = ET.parse(xml_path)
                    root = tree.getroot()

                    objects = root.findall("object")

                    for obj in objects:

                        name = obj.find("name")

                        if name is not None:

                            class_name = name.text.strip()

                            voc_class_counter[
                                class_name
                            ] += 1

                        bbox = obj.find("bndbox")

                        if bbox is not None:

                            voc_boxes += 1

                except Exception:

                    voc_invalid += 1

            add(
                f"Total VOC bounding boxes: "
                f"{voc_boxes}"
            )

            add(
                f"Invalid XML files: "
                f"{voc_invalid}"
            )

            add()

            if voc_class_counter:

                add("VOC class distribution:")

                for name, count in (
                    voc_class_counter.most_common()
                ):

                    add(
                        f"  {name:<30} : {count}"
                    )

        except Exception as e:

            add(
                f"XML analysis error: {e}"
            )

    add()

    # ========================================================
    # IMAGE-LABEL MATCHING
    # ========================================================

    add("=" * 80)
    add("10. IMAGE ↔ LABEL MATCHING")
    add("=" * 80)

    image_stems = defaultdict(list)

    for image in image_files:

        image_stems[
            image.stem.lower()
        ].append(image)

    label_stems = defaultdict(list)

    for label in txt_files:

        label_stems[
            label.stem.lower()
        ].append(label)

    missing_labels = []
    orphan_labels = []

    for stem, images in image_stems.items():

        if stem not in label_stems:

            missing_labels.extend(images)

    for stem, labels in label_stems.items():

        if stem not in image_stems:

            orphan_labels.extend(labels)

    add(
        f"Images without matching TXT label : "
        f"{len(missing_labels)}"
    )

    add(
        f"Labels without matching image      : "
        f"{len(orphan_labels)}"
    )

    if missing_labels:

        add()
        add("Images without labels:")

        for path in missing_labels:

            add(
                f"  {safe_relative(path)}"
            )

    if orphan_labels:

        add()
        add("Orphan label files:")

        for path in orphan_labels:

            add(
                f"  {safe_relative(path)}"
            )

    add()

    # ========================================================
    # DUPLICATE FILE ANALYSIS
    # ========================================================

    add("=" * 80)
    add("11. DUPLICATE FILE ANALYSIS")
    add("=" * 80)

    hash_groups = defaultdict(list)

    for file in all_files:

        digest = file_hash(file)

        if digest:

            hash_groups[digest].append(file)

    duplicate_groups = [
        files
        for files in hash_groups.values()
        if len(files) > 1
    ]

    duplicate_file_count = sum(
        len(group)
        for group in duplicate_groups
    )

    add(
        f"Duplicate groups: "
        f"{len(duplicate_groups)}"
    )

    add(
        f"Files involved in duplicates: "
        f"{duplicate_file_count}"
    )

    if duplicate_groups:

        add()

        for index, group in enumerate(
            duplicate_groups,
            start=1
        ):

            add(
                f"Duplicate group {index}:"
            )

            for file in group:

                add(
                    f"  {safe_relative(file)}"
                )

            add()

    # ========================================================
    # EMPTY FILE ANALYSIS
    # ========================================================

    add("=" * 80)
    add("12. EMPTY / ZERO-BYTE FILES")
    add("=" * 80)

    empty_files = []

    for file in all_files:

        try:

            if file.stat().st_size == 0:

                empty_files.append(file)

        except Exception:

            pass

    add(
        f"Zero-byte files: "
        f"{len(empty_files)}"
    )

    if empty_files:

        for file in empty_files:

            add(
                f"  {safe_relative(file)}"
            )

    add()

    # ========================================================
    # DATASET SPLIT ANALYSIS
    # ========================================================

    add("=" * 80)
    add("13. DATASET SPLIT ANALYSIS")
    add("=" * 80)

    split_keywords = {
        "train",
        "training",
        "val",
        "valid",
        "validation",
        "test",
        "testing"
    }

    split_counts = Counter()

    for file in image_files:

        parts = [
            p.lower()
            for p in file.relative_to(
                DATASET_PATH
            ).parts
        ]

        detected_split = None

        for part in parts:

            if part in split_keywords:

                if part in {"train", "training"}:
                    detected_split = "TRAIN"

                elif part in {
                    "val",
                    "valid",
                    "validation"
                }:
                    detected_split = "VAL"

                elif part in {
                    "test",
                    "testing"
                }:
                    detected_split = "TEST"

                break

        if detected_split:

            split_counts[
                detected_split
            ] += 1

    if split_counts:

        total_split_images = sum(
            split_counts.values()
        )

        for split, count in split_counts.items():

            percentage = (
                count /
                total_split_images *
                100
            )

            add(
                f"{split:<8} : "
                f"{count:<8} "
                f"({percentage:.2f}%)"
            )

    else:

        add(
            "No standard train/val/test split "
            "folders detected."
        )

    add()

    # ========================================================
    # FOLDER-WISE IMAGE COUNTS
    # ========================================================

    add("=" * 80)
    add("14. FOLDER-WISE IMAGE DISTRIBUTION")
    add("=" * 80)

    folder_image_counts = Counter()

    for image in image_files:

        try:

            relative_parent = image.parent.relative_to(
                DATASET_PATH
            )

            folder_name = (
                str(relative_parent)
                if str(relative_parent) != "."
                else "[ROOT]"
            )

            folder_image_counts[
                folder_name
            ] += 1

        except Exception:

            pass

    for folder, count in (
        folder_image_counts.most_common()
    ):

        add(
            f"{folder:<60} : {count}"
        )

    add()

    # ========================================================
    # OTHER FILES
    # ========================================================

    add("=" * 80)
    add("15. OTHER FILES")
    add("=" * 80)

    if other_files:

        for file in other_files:

            try:
                size = format_size(
                    file.stat().st_size
                )
            except Exception:
                size = "Unknown"

            add(
                f"{safe_relative(file)} "
                f"({size})"
            )

    else:

        add("No other files detected.")

    add()

    # ========================================================
    # ARCHIVES
    # ========================================================

    add("=" * 80)
    add("16. ARCHIVE FILES")
    add("=" * 80)

    if archive_files:

        for file in archive_files:

            try:
                size = format_size(
                    file.stat().st_size
                )
            except Exception:
                size = "Unknown"

            add(
                f"{safe_relative(file)} "
                f"({size})"
            )

    else:

        add("No archive files detected.")

    add()

    # ========================================================
    # FINAL DATASET SUMMARY
    # ========================================================

    add("=" * 80)
    add("17. FINAL DATASET SUMMARY")
    add("=" * 80)

    add(f"Dataset path      : {DATASET_PATH}")
    add(f"Directories       : {total_directories}")
    add(f"Total files       : {total_files}")
    add(f"Dataset size      : {format_size(total_size)}")
    add(f"Images            : {len(image_files)}")
    add(f"Annotation files  : {len(annotation_files)}")
    add()

    if yolo_class_counter:

        add("YOLO DATASET:")

        add(
            f"  Classes detected : "
            f"{len(yolo_class_counter)}"
        )

        add(
            f"  Bounding boxes   : "
            f"{total_yolo_boxes}"
        )

        add(
            f"  Empty labels     : "
            f"{empty_label_files}"
        )

        add(
            f"  Invalid labels   : "
            f"{invalid_yolo_lines}"
        )

    add()

    # ========================================================
    # AUTOMATIC DATASET STATUS
    # ========================================================

    add("=" * 80)
    add("18. DATASET HEALTH CHECK")
    add("=" * 80)

    problems = []

    if not image_files:
        problems.append(
            "No image files detected."
        )

    if not annotation_files:
        problems.append(
            "No annotation/config files detected."
        )

    if missing_labels:
        problems.append(
            f"{len(missing_labels)} images "
            f"do not have matching TXT labels."
        )

    if orphan_labels:
        problems.append(
            f"{len(orphan_labels)} TXT labels "
            f"do not have matching images."
        )

    if empty_files:
        problems.append(
            f"{len(empty_files)} zero-byte files detected."
        )

    if duplicate_groups:
        problems.append(
            f"{len(duplicate_groups)} duplicate "
            f"file groups detected."
        )

    if invalid_yolo_lines:
        problems.append(
            f"{invalid_yolo_lines} invalid YOLO "
            f"annotation lines detected."
        )

    if problems:

        add("Potential issues found:")

        for number, problem in enumerate(
            problems,
            start=1
        ):

            add(
                f"{number}. {problem}"
            )

    else:

        add(
            "No major structural problems detected."
        )

    add()

    # ========================================================
    # END
    # ========================================================

    add("=" * 80)
    add("END OF DATASET ANALYSIS")
    add("=" * 80)

    # ========================================================
    # WRITE REPORT
    # ========================================================

    try:

        with open(
            OUTPUT_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                "\n".join(report)
            )

        print()
        print("=" * 70)
        print("DATASET ANALYSIS COMPLETED")
        print("=" * 70)
        print()
        print("Report created:")
        print(OUTPUT_FILE)
        print()

    except Exception as e:

        print(
            f"Could not create report: {e}"
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    analyze_dataset()