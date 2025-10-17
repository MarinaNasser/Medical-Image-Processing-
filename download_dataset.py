import os
import tarfile
import shutil
from pathlib import Path
import requests


DATA_URL = "http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz"
DATA_DIR = Path("./BreakHis_v1")
ARCHIVE_PATH = DATA_DIR / os.path.basename(DATA_URL)
EXTRACTED_DIR = DATA_DIR / "BreakHis_v1"
SELECTED_DIR = DATA_DIR / "BreaKHis_40x_200x"
MAGNIFICATIONS = ["40X", "200X"]


def download_dataset(url=DATA_URL, dest_path=ARCHIVE_PATH):
    """
    Download the BreaKHis dataset if not already downloaded.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        print(f"Dataset archive already exists: {dest_path}")
        return dest_path

    print(f"Downloading dataset from {url} ...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dest_path, "wb") as f:
                downloaded = 0
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        percent = downloaded * 100 / total if total else 0
                        print(f"\rProgress: {percent:5.1f}%", end="")
        print(f"\nDownload complete: {dest_path}")
        return dest_path
    except Exception as e:
        raise RuntimeError(f"Failed to download dataset. Error: {e}")


def extract_dataset(archive_path: Path, extract_to: Path):
    """
    Extract the downloaded tar.gz archive.
    """
    if extract_to.exists():
        print(f"Extraction folder already exists: {extract_to}")
        return extract_to

    print(f"Extracting {archive_path} to {extract_to} ...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_to)
    print("Extraction complete.")
    return extract_to


def filter_magnifications(source_dir: Path, dest_dir: Path, mags=MAGNIFICATIONS):
    """
    Copy only the desired magnifications (40X and 200X) to a separate folder.
    """
    if dest_dir.exists():
        print(f"✅ Filtered folder already exists: {dest_dir}")
        return dest_dir

    print(f"🧩 Filtering magnifications: {', '.join(mags)} ...")
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Walk through the dataset and copy matching magnification folders
    count = 0
    for mag in mags:
        for mag_dir in source_dir.rglob(f"*{mag}*"):
            if mag_dir.is_dir():
                rel_path = mag_dir.relative_to(source_dir)
                target_path = dest_dir / rel_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(mag_dir, target_path, dirs_exist_ok=True)
                count += 1

    print(f"✅ Done. {count} magnification folders copied to {dest_dir}")
    return dest_dir


# ---------------------------
# Main pipeline
# ---------------------------

if __name__ == "__main__":

    # Step 1: Download
    archive = download_dataset()

    # Step 2: Extract
    extracted_path = extract_dataset(archive, EXTRACTED_DIR)

    # Step 3: Filter 40x and 200x only
    filtered_path = filter_magnifications(extracted_path, SELECTED_DIR)

    print(f"   → Archive: {archive}")
    print(f"   → Extracted: {extracted_path}")
    print(f"   → Filtered (40x + 200x): {filtered_path}")
