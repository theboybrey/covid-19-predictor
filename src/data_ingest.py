"""Download or copy the Ghana COVID-19 dataset into `data/raw/`.

This script supports two modes:
1. Kaggle mode (default) - downloads the dataset using the Kaggle API.
2. Manual copy mode - use `--source /path/to/folder` to copy files from a
   local folder (handy if you downloaded `kaggle.json` manually or prefer
   not to use the API).

Usage examples:
    # Kaggle (requires ~/.kaggle/kaggle.json or env vars)
    python src/data_ingest.py

    # Manual: copy all files from a local folder into data/raw/
    python src/data_ingest.py --source ~/Downloads/ghana-covid

"""
import os
import zipfile
import shutil
import argparse
from typing import Optional


DATASET = "kuleafenujoachim/ghana-covid19-cases-dataset"
DEST = os.path.join("data", "raw")


def copy_from_source(source_dir: str, dest: str = DEST):
    """Copy all files from source_dir into dest. Create dest if needed."""
    source_dir = os.path.expanduser(source_dir)
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    os.makedirs(dest, exist_ok=True)
    copied = []
    for fname in os.listdir(source_dir):
        src_path = os.path.join(source_dir, fname)
        if os.path.isfile(src_path):
            dst_path = os.path.join(dest, fname)
            print(f"Copying {src_path} -> {dst_path}")
            shutil.copy2(src_path, dst_path)
            copied.append(dst_path)
            # if it's a zip, try to extract
            if fname.lower().endswith('.zip'):
                try:
                    with zipfile.ZipFile(dst_path, 'r') as z:
                        print(f"Extracting {dst_path}...")
                        z.extractall(dest)
                    os.remove(dst_path)
                except zipfile.BadZipFile:
                    print(f"Warning: {dst_path} is not a valid zip file.")
    print("Copy complete. Files are in:", os.path.abspath(dest))
    return copied


def download_and_extract(dataset: str = DATASET, dest: str = DEST, force: bool = False, kaggle_available: Optional[bool] = None):
    """Download dataset from Kaggle and extract zip files.

    If the Kaggle client is not available, raises ImportError.
    """
    # import locally to avoid hard dependency when using --source path
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
    except Exception as e:
        raise ImportError("Kaggle client not available. Install `kaggle` or use --source.") from e

    os.makedirs(dest, exist_ok=True)
    api = KaggleApi()
    api.authenticate()

    # list files in dataset
    files = api.dataset_list_files(dataset).files
    for f in files:
        fname = f.name
        print(f"Downloading {fname}...")
        api.dataset_download_file(dataset, fname, path=dest, force=force)
        zip_path = os.path.join(dest, fname + ".zip")
        if os.path.exists(zip_path):
            try:
                with zipfile.ZipFile(zip_path, "r") as z:
                    print(f"Extracting {zip_path}...")
                    z.extractall(dest)
                os.remove(zip_path)
            except zipfile.BadZipFile:
                print(f"Warning: {zip_path} is not a zip file or is corrupted. Leaving as-is.")

    print("Download complete. Files are in:", os.path.abspath(dest))


def list_remote_files(dataset: str = DATASET):
    """List files available in the Kaggle dataset (requires kaggle installed)."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
    except Exception as e:
        raise ImportError("Kaggle client not available. Install `kaggle` or use --source.") from e

    api = KaggleApi()
    api.authenticate()
    files = api.dataset_list_files(dataset).files
    print(f"Files available in dataset {dataset}:")
    for f in files:
        print(" -", f.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Ghana COVID-19 dataset from Kaggle or local source")
    parser.add_argument("--source", "-s", help="Local folder to copy files from (skips Kaggle)")
    parser.add_argument("--list", action="store_true", help="List files available on the Kaggle dataset and exit")
    parser.add_argument("--force", action="store_true", help="Force re-download of files (passed to Kaggle API)")
    args = parser.parse_args()

    if args.source:
        try:
            copy_from_source(args.source)
        except Exception as e:
            print("Error copying from source:", e)
    elif args.list:
        try:
            list_remote_files()
        except Exception as e:
            print("Error listing remote files:", e)
            print("If you don't have the Kaggle client installed, run: python -m pip install kaggle")
    else:
        try:
            download_and_extract(force=args.force)
        except ImportError as e:
            print(e)
            print("Either install the Kaggle client in your venv (python -m pip install kaggle) or run with --source /path/to/files")
        except Exception as e:
            print("Download failed:", e)



