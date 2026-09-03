from pathlib import Path

LABEL_DIR = Path("datasets/road_damage/labels/train")

total_files_modified = 0
total_duplicates_removed = 0

print("=" * 70)
print("REMOVING DUPLICATE YOLO LABELS")
print("=" * 70)

for label_file in LABEL_DIR.glob("*.txt"):

    lines = label_file.read_text().splitlines()

    # Preserve order while removing exact duplicates
    unique_lines = list(dict.fromkeys(lines))

    duplicates = len(lines) - len(unique_lines)

    if duplicates > 0:
        label_file.write_text(
            "\n".join(unique_lines) + "\n"
        )

        print(f"{label_file.name}: removed {duplicates} duplicate label(s)")

        total_files_modified += 1
        total_duplicates_removed += duplicates

print("\n" + "=" * 70)
print("CLEANUP COMPLETE")
print("=" * 70)
print(f"Files modified      : {total_files_modified}")
print(f"Duplicates removed  : {total_duplicates_removed}")