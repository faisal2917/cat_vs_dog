"""Download the Microsoft Cats vs Dogs dataset, drop broken images, shrink them,
and split them into train/val/test folders ready for training.

Usage:
    python prepare_data.py                    # download (~825 MB) + prepare
    python prepare_data.py --zip file.zip     # use a zip you already downloaded
    python prepare_data.py --src some_folder  # use already-extracted images

--src works with the Microsoft layout (Cat/, Dog/), the Kaggle layout
(cat.123.jpg, dog.456.jpg) and your own photos in cat(s)/ and dog(s)/ folders.
"""
import argparse
import random
import shutil
import sys
import urllib.request
import warnings
import zipfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from PIL import Image
from tqdm import tqdm

URL = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
DATA_DIR = Path(__file__).resolve().parent / "data"
CLASSES = ("cat", "dog")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def download(url, dest):
    if dest.exists():
        print(f"Found {dest}, skipping download")
        return
    part = dest.with_suffix(".part")
    with urllib.request.urlopen(url) as resp, open(part, "wb") as f, tqdm(
        total=int(resp.headers.get("Content-Length", 0)), unit="B", unit_scale=True, desc="Downloading"
    ) as bar:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            bar.update(len(chunk))
    part.rename(dest)


def extract(zip_path, dest):
    if dest.exists():
        print(f"Found {dest}, skipping extraction")
        return
    part = dest.with_name(dest.name + ".part")
    shutil.rmtree(part, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as z:
        for member in tqdm(z.infolist(), desc="Extracting", unit="file"):
            z.extract(member, part)
    part.rename(dest)


def label_of(path):
    """Label from a cat/dog parent folder, or from a cat./dog. filename prefix (Kaggle)."""
    for name in (path.parent.name, path.name.split(".")[0]):
        name = name.lower().rstrip("s")
        if name in CLASSES:
            return name
    return None


def process(job):
    """Convert one image to an RGB JPEG whose shorter side is at most `size`. False if it is broken."""
    src, dst, size = job
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the dataset has many images with harmless EXIF warnings
            with Image.open(src) as im:
                im = im.convert("RGB")
        scale = size / min(im.size)
        if scale < 1:
            im = im.resize((round(im.width * scale), round(im.height * scale)), Image.Resampling.BICUBIC)
        im.save(dst, "JPEG", quality=90)
        return True
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zip", type=Path, help="already-downloaded dataset zip")
    p.add_argument("--src", type=Path, help="folder of already-extracted images (skips download)")
    p.add_argument("--size", type=int, default=256, help="shrink so the shorter side is this many pixels")
    p.add_argument("--val", type=float, default=0.1, help="fraction of images for validation")
    p.add_argument("--test", type=float, default=0.1, help="fraction of images for the final test")
    p.add_argument("--workers", type=int, default=4, help="parallel processes")
    args = p.parse_args()

    src = args.src
    if src is None:
        DATA_DIR.mkdir(exist_ok=True)
        zip_path = args.zip or DATA_DIR / "kagglecatsanddogs_5340.zip"
        if args.zip is None:
            download(URL, zip_path)
        src = DATA_DIR / "raw"
        extract(zip_path, src)

    files = [(f, label_of(f)) for f in sorted(src.rglob("*")) if f.suffix.lower() in IMG_EXTS]
    files = [(f, label) for f, label in files if label]
    if not files:
        sys.exit(f"No cat/dog images found under {src}")

    out = DATA_DIR / "processed"
    shutil.rmtree(out, ignore_errors=True)
    rng = random.Random(42)
    jobs = []
    for label in CLASSES:
        paths = [f for f, l in files if l == label]
        rng.shuffle(paths)
        n_test, n_val = int(len(paths) * args.test), int(len(paths) * args.val)
        splits = {"test": paths[:n_test], "val": paths[n_test:n_test + n_val], "train": paths[n_test + n_val:]}
        for split, split_paths in splits.items():
            folder = out / split / label
            folder.mkdir(parents=True, exist_ok=True)
            jobs += [(f, folder / f"{label}_{i:05d}.jpg", args.size) for i, f in enumerate(split_paths)]

    with Pool(args.workers) as pool:
        ok = list(tqdm(pool.imap(process, jobs, chunksize=32), total=len(jobs), desc="Cleaning + resizing"))

    print(f"\nSkipped {ok.count(False)} broken images. Ready in {out}:")
    counts = Counter((dst.parent.parent.name, dst.parent.name) for (_, dst, _), good in zip(jobs, ok) if good)
    for split in ("train", "val", "test"):
        print(f"  {split:5}  " + "  ".join(f"{c}: {counts[(split, c)]:>6}" for c in CLASSES))


if __name__ == "__main__":
    main()
