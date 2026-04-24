"""
ingest.py
---------
Download Kinetics-400 videos directly from CVDF S3 hosting.
Videos are pre-cut, pre-hosted tar.gz files — no YouTube scraping,
no dead links, no rate limits.

S3 layout:
  s3://kinetics/400/val/   → k400_val_path.txt lists all tar.gz URLs
  Each tar.gz contains ~50 pre-trimmed MP4 clips for one action class.

Strategy:
  - Fetch the path list from S3 (plain HTTP, no AWS credentials needed)
  - Filter to --limit tar files (each ~50 clips → limit*50 total videos)
  - Download + extract in parallel using ThreadPoolExecutor
  - All output goes to $SCRATCH_DIR/raw_videos/<label>/

Why this is better than yt-dlp:
  - Pre-cut clips (no ffmpeg trim needed)
  - Stable S3 URLs (no YouTube dead links / age gates)
  - ~3–5x faster download (S3 vs YouTube throttling)
  - Directly mirrors real production pipelines (data lake on S3/GCS)

Usage:
  python ingest.py --split val --out_dir $SCRATCH_DIR/raw_videos --limit 10
  python ingest.py --split train --out_dir $SCRATCH_DIR/raw_videos --limit 50
"""

import argparse
import os
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

S3_BASE        = "https://s3.amazonaws.com/kinetics/400"
VAL_PATH_LIST  = f"{S3_BASE}/val/k400_val_path.txt"
TRAIN_PATH_LIST = f"{S3_BASE}/train/k400_train_path.txt"
TEST_PATH_LIST  = f"{S3_BASE}/test/k400_test_path.txt"

PATH_LISTS = {
    "val":   VAL_PATH_LIST,
    "train": TRAIN_PATH_LIST,
    "test":  TEST_PATH_LIST,
}

VAL_CSV_URL   = f"{S3_BASE}/annotations/val.csv"
TRAIN_CSV_URL = f"{S3_BASE}/annotations/train.csv"


def fetch_path_list(split: str) -> list[str]:
    """Fetch the S3 tar.gz URL list for a given split."""
    url = PATH_LISTS[split]
    print(f"Fetching path list: {url}")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    urls = [line.strip() for line in r.text.splitlines() if line.strip()]
    print(f"  Found {len(urls)} tar files for split='{split}'")
    return urls


def download_and_extract(tar_url: str, out_dir: Path) -> dict:
    """
    Download one tar.gz from S3 and extract its MP4s into out_dir/<label>/.

    K400 tar layout: each part_N.tar.gz contains clips from multiple classes.
    Each clip is at <label>/<clip_id>.mp4 inside the tar.
    We preserve the label subdirectory from inside the tar.

    Returns {"url": ..., "status": "ok"|"failed"|"cached", "n_clips": int}
    """
    part_name  = Path(tar_url).stem.replace(".tar", "")   # e.g. "part_0"
    done_marker = out_dir / f".{part_name}.done"

    # Skip if already extracted
    if done_marker.exists():
        n = sum(1 for _ in out_dir.rglob("*.mp4"))
        return {"url": tar_url, "status": "cached", "n_clips": n, "label": part_name}

    tmp_path = None
    t0 = time.perf_counter()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)

        # Stream download into a temp file
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = tmp.name
            r = requests.get(tar_url, stream=True, timeout=300)
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                tmp.write(chunk)

        # Extract — preserve label subdirectory structure: <label>/<clip>.mp4
        n_clips = 0
        with tarfile.open(tmp_path, "r:gz") as tf:
            for m in tf.getmembers():
                if not m.name.endswith(".mp4"):
                    continue
                parts = Path(m.name).parts
                # Expected: <label>/<clip_id>.mp4  (2 parts)
                # Fallback: flat <clip_id>.mp4     (1 part)
                if len(parts) >= 2:
                    label    = parts[-2]
                    clip_name = parts[-1]
                else:
                    label    = "unknown"
                    clip_name = parts[-1]

                dest = out_dir / label
                dest.mkdir(parents=True, exist_ok=True)
                m.name = clip_name   # extract flat into dest
                tf.extract(m, path=dest)
                n_clips += 1

        done_marker.touch()
        elapsed = time.perf_counter() - t0
        return {"url": tar_url, "status": "ok",
                "n_clips": n_clips, "label": part_name, "time_s": elapsed}

    except Exception as e:
        return {"url": tar_url, "status": f"failed: {e}", "n_clips": 0, "label": part_name}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def download_annotation_csv(split: str, out_dir: Path) -> Path:
    """Download the annotation CSV from S3 into out_dir."""
    url      = f"{S3_BASE}/annotations/{split}.csv"
    out_path = out_dir / f"kinetics400_{split}.csv"
    if out_path.exists():
        print(f"  CSV already exists: {out_path}")
        return out_path
    print(f"  Downloading annotation CSV: {url}")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    out_path.write_text(r.text)
    print(f"  Saved → {out_path}  ({len(r.text.splitlines())} rows)")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split",   default="val", choices=["train", "val", "test"],
                    help="Kinetics-400 split to download")
    ap.add_argument("--out_dir", default=None,
                    help="Output directory. Defaults to $SCRATCH_DIR/raw_videos")
    ap.add_argument("--limit",   type=int, default=10,
                    help="Max number of tar.gz files to download (~50 clips each)")
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel download threads")
    args = ap.parse_args()

    scratch  = os.environ.get("SCRATCH_DIR", "data")
    out_dir  = Path(args.out_dir) if args.out_dir else Path(scratch) / "raw_videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output dir : {out_dir}")
    print(f"Split      : {args.split}")
    print(f"Tar limit  : {args.limit}  (~{args.limit * 50} clips)")
    print(f"Workers    : {args.workers}")

    # Download annotation CSV alongside videos
    download_annotation_csv(args.split, Path(scratch))

    # Fetch URL list and cap at limit
    tar_urls = fetch_path_list(args.split)[: args.limit]

    t0 = time.perf_counter()
    ok = failed = cached = total_clips = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(download_and_extract, url, out_dir): url
                   for url in tar_urls}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if   res["status"] == "ok":     ok     += 1; total_clips += res["n_clips"]
            elif res["status"] == "cached": cached += 1; total_clips += res["n_clips"]
            else:                           failed += 1
            if i % 5 == 0 or i == len(tar_urls):
                elapsed = time.perf_counter() - t0
                print(f"  [{i}/{len(tar_urls)}]  ok={ok}  cached={cached}  "
                      f"failed={failed}  clips={total_clips}  {elapsed:.1f}s")

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  ok={ok}  cached={cached}  failed={failed}")
    print(f"  Total clips: {total_clips}")
    print(f"  Output: {out_dir}")


if __name__ == "__main__":
    main()
