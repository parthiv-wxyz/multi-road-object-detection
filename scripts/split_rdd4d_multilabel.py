import shutil
from pathlib import Path
from collections import Counter

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DATASET = PROJECT_ROOT / "datasets" / "road_damage"
TARGET_DATASET = PROJECT_ROOT / "datasets" / "road_damage_v2"


# ============================================================
# SETTINGS
# ============================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SOURCE_SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}

CLASS_NAMES = [
    "Alligator",
    "Block",
    "Longitudinal",
    "Transversal",
    "Pot Hole",
]

NUM_CLASSES = len(CLASS_NAMES)


# ============================================================
# GET IMAGE-LABEL PAIRS
# ============================================================

def get_image_label_pairs():

    pairs = []

    for source_split in SOURCE_SPLITS:

        image_dir = (
            SOURCE_DATASET /
            "images" /
            source_split
        )

        label_dir = (
            SOURCE_DATASET /
            "labels" /
            source_split
        )

        if not image_dir.exists():

            print(
                f"WARNING: Image directory not found: "
                f"{image_dir}"
            )

            continue

        if not label_dir.exists():

            raise FileNotFoundError(
                f"Label directory not found: "
                f"{label_dir}"
            )

        image_files = sorted(
            [
                path
                for path in image_dir.iterdir()
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in IMAGE_EXTENSIONS
                )
            ]
        )

        print(
            f"\nScanning {source_split.upper()}: "
            f"{len(image_files)} images"
        )

        for image_path in image_files:

            label_path = (
                label_dir /
                f"{image_path.stem}.txt"
            )

            if not label_path.exists():

                raise FileNotFoundError(
                    f"\nMissing label file:\n"
                    f"Image: {image_path}\n"
                    f"Expected label: {label_path}"
                )

            pairs.append(
                (
                    source_split,
                    image_path,
                    label_path
                )
            )

    return pairs


# ============================================================
# CREATE MULTILABEL VECTOR
# ============================================================

def get_multilabel_vector(label_path):

    vector = np.zeros(
        NUM_CLASSES,
        dtype=int
    )

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            parts = line.strip().split()

            if not parts:
                continue

            try:

                class_id = int(parts[0])

            except ValueError:

                raise ValueError(
                    f"Invalid class ID in "
                    f"{label_path}: {line}"
                )

            if 0 <= class_id < NUM_CLASSES:

                vector[class_id] = 1

            else:

                raise ValueError(
                    f"Invalid class ID {class_id} "
                    f"in {label_path}"
                )

    return vector


# ============================================================
# COUNT OBJECT INSTANCES
# ============================================================

def count_instances(label_paths):

    counts = Counter()

    for label_path in label_paths:

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                parts = line.strip().split()

                if not parts:
                    continue

                class_id = int(parts[0])

                if 0 <= class_id < NUM_CLASSES:

                    counts[class_id] += 1

    return counts


# ============================================================
# PREPARE TARGET DIRECTORIES
# ============================================================

def prepare_directories():

    if TARGET_DATASET.exists():

        print("\nERROR: Target dataset already exists:")
        print(TARGET_DATASET)

        print(
            "\nDelete or rename the existing "
            "dataset before running this script."
        )

        raise SystemExit(1)

    for split in SOURCE_SPLITS:

        image_dir = (
            TARGET_DATASET /
            "images" /
            split
        )

        label_dir = (
            TARGET_DATASET /
            "labels" /
            split
        )

        image_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        label_dir.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# COPY SPLIT
# ============================================================

def copy_split(
    indices,
    split_name,
    images,
    labels,
    source_splits
):

    copied_images = []
    copied_labels = []

    for index in indices:

        image_path = images[index]
        label_path = labels[index]

        original_split = (
            source_splits[index]
        )

        # ----------------------------------------------------
        # CREATE UNIQUE FILENAME
        #
        # Example:
        #
        # train_1001.JPG
        # val_1001.JPG
        # test_1001.JPG
        #
        # This prevents files with identical names from
        # overwriting each other.
        # ----------------------------------------------------

        unique_stem = (
            f"{original_split}_"
            f"{image_path.stem}"
        )

        target_image = (
            TARGET_DATASET /
            "images" /
            split_name /
            f"{unique_stem}{image_path.suffix}"
        )

        target_label = (
            TARGET_DATASET /
            "labels" /
            split_name /
            f"{unique_stem}.txt"
        )

        # ----------------------------------------------------
        # SAFETY CHECK
        # ----------------------------------------------------

        if target_image.exists():

            raise RuntimeError(
                f"\nDuplicate output image detected:\n"
                f"{target_image}"
            )

        if target_label.exists():

            raise RuntimeError(
                f"\nDuplicate output label detected:\n"
                f"{target_label}"
            )

        # ----------------------------------------------------
        # COPY FILES
        # ----------------------------------------------------

        shutil.copy2(
            image_path,
            target_image
        )

        shutil.copy2(
            label_path,
            target_label
        )

        copied_images.append(
            target_image
        )

        copied_labels.append(
            target_label
        )

    return copied_images, copied_labels


# ============================================================
# PRINT SPLIT SUMMARY
# ============================================================

def print_summary(
    split_name,
    images,
    labels
):

    instance_counts = count_instances(
        labels
    )

    image_class_counts = Counter()

    for label_path in labels:

        present_classes = set()

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                parts = line.strip().split()

                if not parts:
                    continue

                class_id = int(parts[0])

                if (
                    0 <= class_id
                    < NUM_CLASSES
                ):

                    present_classes.add(
                        class_id
                    )

        for class_id in present_classes:

            image_class_counts[
                class_id
            ] += 1

    print("\n" + "=" * 65)
    print(split_name.upper())
    print("=" * 65)

    print(
        f"Images: {len(images)}"
    )

    print(
        f"Labels: {len(labels)}"
    )

    print(
        "\nUnique images containing each class:"
    )

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"  {class_id} -> "
            f"{class_name:<15}: "
            f"{image_class_counts[class_id]}"
        )

    print("\nObject instances:")

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"  {class_id} -> "
            f"{class_name:<15}: "
            f"{instance_counts[class_id]}"
        )


# ============================================================
# MAIN
# ============================================================

print("=" * 65)
print("RDD4D MULTILABEL STRATIFIED DATASET SPLIT")
print("=" * 65)

print("\nSource dataset:")
print(SOURCE_DATASET)

print("\nTarget dataset:")
print(TARGET_DATASET)


# ============================================================
# LOAD IMAGE-LABEL PAIRS
# ============================================================

pairs = get_image_label_pairs()

if not pairs:

    raise FileNotFoundError(
        f"No image-label pairs found in:\n"
        f"{SOURCE_DATASET}"
    )


source_splits = [
    pair[0]
    for pair in pairs
]

images = [
    pair[1]
    for pair in pairs
]

labels = [
    pair[2]
    for pair in pairs
]


print(
    f"\nTotal image-label pairs: "
    f"{len(images)}"
)


# ============================================================
# CREATE MULTILABEL MATRIX
# ============================================================

print(
    "\nCreating multilabel matrix..."
)

Y = np.array(
    [
        get_multilabel_vector(
            label_path
        )
        for label_path in labels
    ]
)

X = np.arange(
    len(images)
)


# ============================================================
# FIRST SPLIT
#
# TRAIN = 70%
# TEMP  = 30%
# ============================================================

splitter_1 = (
    MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=(
            VAL_RATIO +
            TEST_RATIO
        ),
        random_state=42
    )
)

train_indices, temp_indices = next(
    splitter_1.split(
        X,
        Y
    )
)


# ============================================================
# SECOND SPLIT
#
# TEMP -> VAL = 15%
# TEMP -> TEST = 15%
# ============================================================

X_temp = X[temp_indices]

Y_temp = Y[temp_indices]

splitter_2 = (
    MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=42
    )
)

val_relative, test_relative = next(
    splitter_2.split(
        X_temp,
        Y_temp
    )
)

val_indices = temp_indices[
    val_relative
]

test_indices = temp_indices[
    test_relative
]


# ============================================================
# VERIFY SPLIT ASSIGNMENT
# ============================================================

all_indices = (
    list(train_indices)
    + list(val_indices)
    + list(test_indices)
)


if len(all_indices) != len(
    set(all_indices)
):

    raise RuntimeError(
        "\nERROR: Duplicate images "
        "detected between splits."
    )


if len(all_indices) != len(
    images
):

    raise RuntimeError(
        "\nERROR: Some images were "
        "not assigned to a split."
    )


print("\nSplit assignment verified.")

print(
    f"Train: {len(train_indices)}"
)

print(
    f"Val  : {len(val_indices)}"
)

print(
    f"Test : {len(test_indices)}"
)


# ============================================================
# CREATE TARGET DIRECTORIES
# ============================================================

prepare_directories()


# ============================================================
# COPY DATASET
# ============================================================

print(
    "\nCreating new dataset..."
)


train_images, train_labels = copy_split(
    train_indices,
    "train",
    images,
    labels,
    source_splits
)


val_images, val_labels = copy_split(
    val_indices,
    "val",
    images,
    labels,
    source_splits
)


test_images, test_labels = copy_split(
    test_indices,
    "test",
    images,
    labels,
    source_splits
)


# ============================================================
# FINAL FILE COUNT VERIFICATION
# ============================================================

total_copied = (
    len(train_images)
    + len(val_images)
    + len(test_images)
)

if total_copied != len(images):

    raise RuntimeError(
        "\nERROR: Image count mismatch "
        "after copying."
    )


print(
    "\nFile copy verification passed."
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("SPLIT COMPLETE")
print("=" * 65)

print(
    f"\nTRAIN: {len(train_images)} images"
)

print(
    f"VAL  : {len(val_images)} images"
)

print(
    f"TEST : {len(test_images)} images"
)


print_summary(
    "Train",
    train_images,
    train_labels
)


print_summary(
    "Validation",
    val_images,
    val_labels
)


print_summary(
    "Test",
    test_images,
    test_labels
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 65)
print(
    "NEW DATASET CREATED SUCCESSFULLY"
)
print("=" * 65)

print(
    f"\nLocation:\n"
    f"{TARGET_DATASET}"
)

print(
    f"\nTotal images: "
    f"{total_copied}"
)