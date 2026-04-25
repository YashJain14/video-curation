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
import csv
import os
import shutil
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import wandb

S3_BASE        = "https://s3.amazonaws.com/kinetics/400"
VAL_PATH_LIST  = f"{S3_BASE}/val/k400_val_path.txt"
TRAIN_PATH_LIST = f"{S3_BASE}/train/k400_train_path.txt"
TEST_PATH_LIST  = f"{S3_BASE}/test/k400_test_path.txt"

PATH_LISTS = {
    "val":   VAL_PATH_LIST,
    "train": TRAIN_PATH_LIST,
    "test":  TEST_PATH_LIST,
}


def fetch_path_list(split: str) -> list[str]:
    """Fetch the S3 tar.gz URL list for a given split."""
    url = PATH_LISTS[split]
    print(f"Fetching path list: {url}")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    urls = [line.strip() for line in r.text.splitlines() if line.strip()]
    print(f"  Found {len(urls)} tar files for split='{split}'")
    return urls


def load_label_lookup(csv_path: Path) -> dict[str, str]:
    """
    Build {youtube_id: label} from the annotation CSV.
    CSV format (no header): label, youtube_id, time_start, time_end, split, index
    Clip filenames in the tar are: <youtube_id>_<start>_<end>.mp4
    """
    lookup = {}
    with open(csv_path) as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            label, yt_id = row[0].strip(), row[1].strip()
            lookup[yt_id] = label.replace(" ", "_")
    return lookup


def download_and_extract(tar_url: str, out_dir: Path,
                         label_lookup: dict[str, str]) -> dict:
    """
    Download one tar.gz from S3 and extract MPs into out_dir/<label>/.

    K400 tar layout: flat ./  with filenames <yt_id>_<start>_<end>.mp4
    Label is derived by matching the yt_id prefix against the annotation CSV.

    Returns {"url": ..., "status": "ok"|"failed"|"cached", "n_clips": int}
    """
    part_name   = Path(tar_url).stem.replace(".tar", "")
    done_marker = out_dir / f".{part_name}.done"

    if done_marker.exists():
        return {"url": tar_url, "status": "cached", "n_clips": 0}

    tmp_path = None
    t0 = time.perf_counter()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = tmp.name
            r = requests.get(tar_url, stream=True, timeout=300)
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=1 << 20):
                tmp.write(chunk)

        n_clips = 0
        with tarfile.open(tmp_path, "r:gz") as tf:
            for m in tf.getmembers():
                if not m.name.endswith(".mp4"):
                    continue
                clip_name = Path(m.name).name          # strip leading ./
                # filename: <yt_id>_<000000>_<000010>.mp4
                # yt_id is everything before the last two _NNNNNN segments
                parts  = clip_name.replace(".mp4", "").rsplit("_", 2)
                yt_id  = parts[0] if len(parts) == 3 else clip_name
                label  = label_lookup.get(yt_id, "unknown")

                dest = out_dir / label
                dest.mkdir(parents=True, exist_ok=True)
                m.name = clip_name
                tf.extract(m, path=dest)
                n_clips += 1

        done_marker.touch()
        elapsed = time.perf_counter() - t0
        return {"url": tar_url, "status": "ok", "n_clips": n_clips, "time_s": elapsed}

    except Exception as e:
        return {"url": tar_url, "status": f"failed: {e}", "n_clips": 0}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def relabel_existing(raw_dir: Path, label_lookup: dict[str, str]) -> int:
    """
    Move clips out of 'unknown/' into correct label folders.
    Used to fix clips extracted before the label lookup was available.
    Returns number of clips moved.
    """
    unknown_dir = raw_dir / "unknown"
    if not unknown_dir.exists():
        return 0
    moved = 0
    for mp4 in list(unknown_dir.glob("*.mp4")):
        parts = mp4.stem.rsplit("_", 2)
        yt_id = parts[0] if len(parts) == 3 else mp4.stem
        label = label_lookup.get(yt_id, "unknown")
        if label == "unknown":
            continue
        dest = raw_dir / label
        dest.mkdir(parents=True, exist_ok=True)
        shutil.move(str(mp4), str(dest / mp4.name))
        moved += 1
    if moved:
        # Clean up unknown dir if empty
        remaining = list(unknown_dir.glob("*.mp4"))
        if not remaining:
            unknown_dir.rmdir()
    return moved


def download_annotation_csv(split: str, scratch: Path) -> Path:
    """Download the annotation CSV from S3 if not already present."""
    out_path = scratch / f"kinetics400_{split}.csv"
    if out_path.exists():
        print(f"  CSV already exists: {out_path}")
        return out_path
    url = f"{S3_BASE}/annotations/{split}.csv"
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

    scratch  = os.environ.get("SCRATCH_DIR") or os.path.expanduser("~/scratch/video-curation")
    out_dir  = Path(args.out_dir) if args.out_dir else Path(scratch) / "raw_videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output dir : {out_dir}")
    print(f"Split      : {args.split}")
    print(f"Tar limit  : {args.limit}")
    print(f"Workers    : {args.workers}")

    wandb.init(project="video-curation", entity="rlx-labs",
               name="ingest-stage", resume="allow",
               config={"split":   args.split,
                       "limit":   args.limit,
                       "workers": args.workers,
                       "out_dir": str(out_dir)})

    csv_path     = download_annotation_csv(args.split, Path(scratch))
    label_lookup = load_label_lookup(csv_path)
    print(f"  Label lookup: {len(label_lookup)} entries")

    # Fix any clips already sitting in unknown/ from a previous run
    moved = relabel_existing(out_dir, label_lookup)
    if moved:
        print(f"  Re-labelled {moved} existing clips from unknown/")

    tar_urls = fetch_path_list(args.split)[: args.limit]

    t0 = time.perf_counter()
    ok = failed = cached = total_clips = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(download_and_extract, url, out_dir, label_lookup): url
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
    n_labels = len(set(p.parent.name for p in out_dir.rglob('*.mp4')))
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  ok={ok}  cached={cached}  failed={failed}")
    print(f"  Total clips: {total_clips}")
    print(f"  Labels found: {n_labels}")
    print(f"  Output: {out_dir}")

    wandb.log({
        "ingest/tars_ok":     ok,
        "ingest/tars_cached": cached,
        "ingest/tars_failed": failed,
        "ingest/total_clips": total_clips,
        "ingest/n_labels":    n_labels,
        "ingest/elapsed_s":   elapsed,
        "ingest/clips_per_s": total_clips / max(elapsed, 1e-6),
    })
    wandb.finish()


if __name__ == "__main__":
    main()
