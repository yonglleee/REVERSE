"""
filter_corrupt_images.py  —  Remove rows from MP16_Pro_fixed.csv whose images
                              cannot be opened by PIL (corrupt files).

Reads IMG_IDs from CSV, constructs paths, validates in parallel, saves clean CSV.

Usage:
    python filter_corrupt_images.py \
        --csv        /mnt/sh/mmvision/home/jonahli/data/MP16-Pro/metadata/MP16_Pro_fixed.csv \
        --images-dir /mnt/sh/mmvision/home/jonahli/data/MP16-Pro/images \
        --workers    32
"""

import argparse
import os
import logging
import multiprocessing as mp

import pandas as pd
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def check_image(path: str) -> tuple:
    """Return (img_id_filename, is_valid).

    Uses img.load() instead of img.verify() so that truncated images (valid
    header but missing pixel bytes) are also caught.  img.verify() only checks
    the file structure/header and silently passes truncated files that later
    raise OSError during actual decode.
    """
    fname = os.path.basename(path)
    try:
        with Image.open(path) as img:
            img.load()   # forces full pixel decode; raises OSError on truncated files
        return fname, True
    except Exception:
        return fname, False


def img_id_to_path(img_id: str, images_dir: str) -> str:
    """e.g. '81_71_1290503561.jpg' -> images_dir/81/71/81_71_1290503561.jpg"""
    parts = img_id.split("_")
    return os.path.join(images_dir, parts[0], parts[1], img_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",
                        default="/mnt/sh/mmvision/home/jonahli/data/MP16-Pro/metadata/MP16_Pro_fixed.csv")
    parser.add_argument("--images-dir",
                        default="/mnt/sh/mmvision/home/jonahli/data/MP16-Pro/images")
    parser.add_argument("--output", default=None,
                        help="Output path. Defaults to overwriting --csv.")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    output_path = args.output or args.csv

    # ── Step 1: load CSV and get unique IMG_IDs ───────────────────────────────
    log.info("Loading CSV: %s", args.csv)
    df = pd.read_csv(args.csv)
    n_before = len(df)
    log.info("Loaded %d rows. Unique IMG_IDs: %d", n_before, df["IMG_ID"].nunique())

    unique_ids = df["IMG_ID"].unique().tolist()

    # ── Step 2: parallel PIL verify ──────────────────────────────────────────
    paths = [img_id_to_path(img_id, args.images_dir) for img_id in unique_ids]
    log.info("Validating %d unique images with %d workers...", len(paths), args.workers)

    bad_ids = set()
    done = 0
    report_every = max(1, len(paths) // 20)

    with mp.Pool(args.workers) as pool:
        for fname, ok in pool.imap_unordered(check_image, paths, chunksize=512):
            done += 1
            if not ok:
                bad_ids.add(fname)
            if done % report_every == 0:
                log.info("  %d / %d validated  (bad so far: %d)", done, len(paths), len(bad_ids))

    log.info("Validation done. good=%d  bad=%d", len(paths) - len(bad_ids), len(bad_ids))

    if bad_ids:
        log.info("Corrupt image IDs:")
        for bid in sorted(bad_ids):
            log.info("  %s", bid)

    # ── Step 3: filter CSV ────────────────────────────────────────────────────
    df_clean = df[~df["IMG_ID"].isin(bad_ids)].copy()
    n_after = len(df_clean)
    n_dropped = n_before - n_after
    log.info("Kept %d rows, dropped %d (%.3f%%) with corrupt images.",
             n_after, n_dropped, 100.0 * n_dropped / n_before if n_before else 0)

    log.info("Saving to %s ...", output_path)
    df_clean.to_csv(output_path, index=False)
    log.info("Done.")


if __name__ == "__main__":
    main()
