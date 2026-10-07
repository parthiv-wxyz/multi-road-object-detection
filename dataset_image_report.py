from pathlib import Path
import csv

# ============================================================
# CONFIGURATION
# ============================================================

# Main folder containing all your datasets
DATASETS_ROOT = Path(
    r"E:\Parthiv\multi-road-object-detection\datasets"
)

# Image extensions
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}

# Output files
TXT_REPORT = DATASETS_ROOT / "all_dataset_image_report.txt"
CSV_REPORT = DATASETS_ROOT / "all_dataset_image_report.csv"


# ============================================================
# COUNT IMAGES IN A FOLDER
# ============================================================

def count_images(folder):
    return sum(
        1
        for file in folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    )


# ============================================================
# GENERATE REPORT
# ============================================================

def generate_report():

    if not DATASETS_ROOT.exists():
        print("ERROR: Dataset root folder not found:")
        print(DATASETS_ROOT)
        return

    txt_lines = []

    csv_rows = []

    grand_total = 0

    txt_lines.append("=" * 80)
    txt_lines.append("COMPLETE DATASET IMAGE REPORT")
    txt_lines.append("=" * 80)
    txt_lines.append(f"Root folder: {DATASETS_ROOT}")
    txt_lines.append("")

    # --------------------------------------------------------
    # Find all dataset folders
    # --------------------------------------------------------

    datasets = sorted(
        [
            folder
            for folder in DATASETS_ROOT.iterdir()
            if folder.is_dir()
        ]
    )

    if not datasets:
        print("No dataset folders found.")
        return

    # --------------------------------------------------------
    # Process every dataset
    # --------------------------------------------------------

    for dataset in datasets:

        print("\n" + "=" * 80)
        print(f"DATASET: {dataset.name}")
        print("=" * 80)

        txt_lines.append("")
        txt_lines.append("=" * 80)
        txt_lines.append(f"DATASET: {dataset.name}")
        txt_lines.append("=" * 80)

        dataset_total = 0

        # Include dataset root itself
        folders = [dataset]

        # Add every subfolder
        folders.extend(
            sorted(
                [
                    folder
                    for folder in dataset.rglob("*")
                    if folder.is_dir()
                ]
            )
        )

        for folder in folders:

            image_count = count_images(folder)

            # Only report folders that contain images
            if image_count == 0:
                continue

            relative_path = folder.relative_to(dataset)

            if str(relative_path) == ".":
                folder_display = "."
            else:
                folder_display = str(relative_path)

            # Console
            print(
                f"{folder_display:<55} : {image_count:>6} images"
            )

            # TXT
            txt_lines.append(
                f"{folder_display:<55} : {image_count:>6} images"
            )

            # CSV
            csv_rows.append([
                dataset.name,
                folder_display,
                image_count
            ])

            dataset_total += image_count

        # Dataset total
        txt_lines.append("-" * 80)
        txt_lines.append(
            f"{dataset.name} TOTAL IMAGES: {dataset_total}"
        )

        print("-" * 80)
        print(
            f"{dataset.name} TOTAL IMAGES: {dataset_total}"
        )

        grand_total += dataset_total

    # --------------------------------------------------------
    # Grand total
    # --------------------------------------------------------

    txt_lines.append("")
    txt_lines.append("=" * 80)
    txt_lines.append(
        f"TOTAL IMAGES IN ALL DATASETS: {grand_total}"
    )
    txt_lines.append("=" * 80)

    print("\n" + "=" * 80)
    print(f"TOTAL IMAGES IN ALL DATASETS: {grand_total}")
    print("=" * 80)

    # ========================================================
    # SAVE TXT
    # ========================================================

    with open(
        TXT_REPORT,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("\n".join(txt_lines))

    # ========================================================
    # SAVE CSV
    # ========================================================

    with open(
        CSV_REPORT,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Dataset",
            "Folder",
            "Image Count"
        ])

        writer.writerows(csv_rows)

    # ========================================================
    # FINISHED
    # ========================================================

    print("\nREPORTS SAVED")
    print("-" * 80)
    print(f"TXT : {TXT_REPORT}")
    print(f"CSV : {CSV_REPORT}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    generate_report()