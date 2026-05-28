import os
from pathlib import Path

# --- CONFIG ---
FOLDER = r"D:\[THESIS]\dataset\DLSU 10-13\RGB"  # folder containing the files to rename
START_NUM = 158                         # starting number (e.g. 1, 50, 100)
EXTENSIONS = None                     # filter by extension e.g. [".jpg", ".png"], or None for all files
PREFIX = ""                           # optional prefix before the number e.g. "img_"
SUFFIX = ""                           # optional suffix after the number, before extension e.g. "_defect"
# --------------

def rename_files(folder, start_num, extensions=None, prefix="", suffix=""):
    folder = Path(folder)
    if not folder.exists():
        print(f"Folder not found: {folder}")
        return

    files = sorted([
        f for f in folder.iterdir()
        if f.is_file() and (extensions is None or f.suffix.lower() in extensions)
    ])

    if not files:
        print("No files found.")
        return

    total = len(files)
    end_num = start_num + total - 1
    pad_width = len(str(end_num))

    print(f"Found {total} files. Renaming from {str(start_num).zfill(pad_width)} to {str(end_num).zfill(pad_width)}.")
    print("-" * 40)

    for i, file in enumerate(files):
        num = start_num + i
        new_name = f"{prefix}{str(num).zfill(pad_width)}{suffix}{file.suffix}"
        new_path = folder / new_name

        if file == new_path:
            print(f"[skip] {file.name} (already correct)")
            continue

        if new_path.exists():
            print(f"[warn] {new_name} already exists, skipping {file.name}")
            continue

        print(f"{file.name}  ->  {new_name}")
        file.rename(new_path)

    print("-" * 40)
    print("Done.")

if __name__ == "__main__":
    rename_files(FOLDER, START_NUM, EXTENSIONS, PREFIX, SUFFIX)
