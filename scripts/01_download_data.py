"""
Step 0 - Download the raw datasets.

Downloads HDFS_v1 and BGL from Loghub (Zenodo record 8196385), checks each
file's MD5 checksum (proves the file is complete and unchanged), and unzips it
into data/raw/<DATASET>/.

Usage:
    python scripts/01_download_data.py            # both datasets
    python scripts/01_download_data.py --only BGL
"""
import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATASETS, RAW_DIR, ZENODO_RECORD  # noqa: E402

URL = "https://zenodo.org/api/records/{record}/files/{name}/content"


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                         desc=dest.name) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=list(DATASETS), help="download one dataset")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, info in DATASETS.items():
        if args.only and name != args.only:
            continue
        if info["log_file"].exists():
            print(f"[{name}] already extracted at {info['log_file']} - skipping")
            continue

        zip_path = RAW_DIR / info["zip_name"]
        if not zip_path.exists() or md5_of(zip_path) != info["md5"]:
            print(f"[{name}] downloading {info['zip_name']} ...")
            download(URL.format(record=ZENODO_RECORD, name=info["zip_name"]), zip_path)

        checksum = md5_of(zip_path)
        if checksum != info["md5"]:
            sys.exit(f"[{name}] MD5 mismatch ({checksum}); delete {zip_path} and retry")
        print(f"[{name}] checksum OK")

        target = RAW_DIR / name
        print(f"[{name}] extracting to {target} ...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(target)
        print(f"[{name}] done")


if __name__ == "__main__":
    main()
