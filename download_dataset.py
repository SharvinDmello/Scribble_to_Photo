#!/usr/bin/env python3
import os
import sys
import time
import shutil
import tarfile
import urllib.request
import argparse
from pathlib import Path

DEFAULT_DATASET_URL = "https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/edges2shoes.tar.gz"


def download_file_with_resume(url: str, dest_path: Path, chunk_size: int = 1024 * 1024):
    if url.startswith("http://efrosgans.eecs.berkeley.edu"):
        url = url.replace("http://", "https://", 1)

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "*/*"
    }

    initial_size = 0
    if dest_path.exists():
        initial_size = dest_path.stat().st_size
        if initial_size > 0:
            headers["Range"] = f"bytes={initial_size}-"
            print(f"[INFO] Found partial download ({initial_size / (1024*1024):.1f} MB). Resuming...")

    req = urllib.request.Request(url, headers=headers)

    try:
        response = urllib.request.urlopen(req, timeout=30)
    except urllib.error.HTTPError as e:
        if e.code == 416 or initial_size > 0:
            print(f"[WARN] Range request returned {e.code}. Restarting full download from 0...")
            if dest_path.exists():
                dest_path.unlink()
            headers.pop("Range", None)
            req = urllib.request.Request(url, headers=headers)
            response = urllib.request.urlopen(req, timeout=30)
        else:
            raise

    content_length = response.headers.get("Content-Length")
    if content_length is not None:
        total_size = int(content_length) + (initial_size if response.status == 206 else 0)
    else:
        total_size = None

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    mode = "ab" if (response.status == 206 and initial_size > 0) else "wb"
    if mode == "wb":
        initial_size = 0

    print(f"[INFO] Downloading dataset from: {url}")
    if total_size:
        print(f"[INFO] Total archive size: {total_size / (1024*1024):.1f} MB")

    if has_tqdm:
        pbar = tqdm(
            total=total_size,
            initial=initial_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Downloading edges2shoes",
            miniters=1
        )
    else:
        pbar = None

    start_time = time.time()
    downloaded = initial_size

    try:
        with open(dest_path, mode) as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                if pbar:
                    pbar.update(len(chunk))
                else:
                    elapsed = max(0.1, time.time() - start_time)
                    speed_mb = (downloaded - initial_size) / (1024 * 1024 * elapsed)
                    if total_size:
                        pct = (downloaded / total_size) * 100.0
                        sys.stdout.write(f"\rDownloading: {pct:.1f}% ({downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB) @ {speed_mb:.2f} MB/s")
                    else:
                        sys.stdout.write(f"\rDownloaded: {downloaded // (1024*1024)}MB @ {speed_mb:.2f} MB/s")
                    sys.stdout.flush()

        if pbar:
            pbar.close()
        else:
            sys.stdout.write("\n")

        print("[INFO] Download completed successfully.")
    except (Exception, KeyboardInterrupt) as err:
        if pbar:
            pbar.close()
        print(f"\n[ERROR] Download interrupted: {err}")
        print(f"[INFO] Partial file preserved at {dest_path}. You can rerun to resume.")
        raise


def extract_tar_archive(archive_path: Path, target_dir: Path):
    print(f"[INFO] Extracting archive {archive_path.name} into {target_dir}...")

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        if has_tqdm:
            for member in tqdm(members, desc="Extracting files", unit="file"):
                tar.extract(member, path=target_dir)
        else:
            print(f"[INFO] Extracting {len(members)} files, please wait...")
            tar.extractall(path=target_dir)

    print("[INFO] Extraction completed successfully.")


def download_and_extract(
    dataset_url: str = DEFAULT_DATASET_URL,
    target_root: str = "datasets",
    clean_archive: bool = False
):
    target_root_path = Path(target_root).resolve()
    target_root_path.mkdir(parents=True, exist_ok=True)

    archive_name = "edges2shoes.tar.gz"
    archive_path = target_root_path / archive_name
    final_extracted_dir = target_root_path / "edges2shoes"

    if final_extracted_dir.exists() and (final_extracted_dir / "train").exists():
        num_train = len(list((final_extracted_dir / "train").glob("*.jpg")))
        num_val = len(list((final_extracted_dir / "val").glob("*.jpg"))) if (final_extracted_dir / "val").exists() else 0
        if num_train > 0:
            print(f"[SUCCESS] Dataset already present at: {final_extracted_dir}")
            print(f"          Found {num_train} training samples and {num_val} validation samples.")
            return str(final_extracted_dir)

    download_file_with_resume(dataset_url, archive_path)
    extract_tar_archive(archive_path, target_root_path)

    if not (final_extracted_dir / "train").exists():
        possible_nested = target_root_path / "edges2shoes" / "edges2shoes"
        if possible_nested.exists():
            for item in possible_nested.iterdir():
                shutil.move(str(item), str(final_extracted_dir))
            shutil.rmtree(str(possible_nested))

    if clean_archive and archive_path.exists():
        print(f"[INFO] Cleaning up {archive_path.name} to save disk space.")
        archive_path.unlink()

    num_train = len(list((final_extracted_dir / "train").glob("*.jpg")))
    num_val = len(list((final_extracted_dir / "val").glob("*.jpg"))) if (final_extracted_dir / "val").exists() else 0

    print(f"\n[SUCCESS] 'edges2shoes' dataset is ready at: {final_extracted_dir}")
    print(f"          Training samples: {num_train}")
    print(f"          Validation samples: {num_val}")
    return str(final_extracted_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract Berkeley edges2shoes Pix2Pix dataset.")
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_DATASET_URL,
        help="URL of the edges2shoes dataset archive."
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="datasets",
        help="Target parent directory for datasets (default: datasets)."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the tar.gz archive after extraction to save disk space."
    )

    args = parser.parse_args()
    download_and_extract(dataset_url=args.url, target_root=args.target_dir, clean_archive=args.clean)
