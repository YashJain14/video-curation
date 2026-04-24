"""
ingest.py
---------
Download a subset of Kinetics-400 clips into a local directory.
Uses yt-dlp to fetch each clip from YouTube given a CSV manifest.

Kinetics-400 CSV format (official splits from DeepMind):
  youtube_id, time_start, time_end, label, split

Usage:
  python ingest.py --csv data/kinetics400_val.csv --out_dir data/raw_videos --limit 500
"""

import argparse
import csv
import subprocess
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def download_clip(row: dict, out_dir: Path) -> dict:
    """Download a single clip trimmed to [time_start, time_end]."""
    yt_id     = row["youtube_id"].strip()
    t_start   = int(row["time_start"])
    t_end     = int(row["time_end"])
    label     = row["label"].strip().replace(" ", "_")
    duration  = t_end - t_start

    label_dir = out_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)

    out_path = label_dir / f"{yt_id}_{t_start:06d}.mp4"
    if out_path.exists():
        return {"yt_id": yt_id, "path": str(out_path), "status": "cached"}

    url = f"https://www.youtube.com/watch?v={yt_id}"
    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "--format", "bestvideo[ext=mp4][height<=480]+bestaudio[ext=m4a]/best[ext=mp4][height<=480]",
        "--download-sections", f"*{t_start}-{t_end}",
        "--force-keyframes-at-cuts",
        "--merge-output-format", "mp4",
        # faststart so DALI can read it
        "--postprocessor-args", "ffmpeg:-movflags +faststart",
        "-o", str(out_path),
        url,
    ]
    try:
        subprocess.run(cmd, check=True, timeout=60,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {"yt_id": yt_id, "path": str(out_path), "status": "ok"}
    except Exception as e:
        return {"yt_id": yt_id, "path": None, "status": f"failed: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",     required=True,  help="Kinetics-400 CSV split file")
    ap.add_argument("--out_dir", default="data/raw_videos")
    ap.add_argument("--limit",   type=int, default=500, help="Max clips to download")
    ap.add_argument("--workers", type=int, default=8,   help="Parallel download threads")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if len(rows) >= args.limit:
                break

    print(f"Downloading {len(rows)} clips → {out_dir}")

    ok = failed = cached = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(download_clip, r, out_dir): r for r in rows}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if   res["status"] == "ok":     ok     += 1
            elif res["status"] == "cached": cached += 1
            else:                           failed += 1
            if i % 50 == 0:
                print(f"  {i}/{len(rows)}  ok={ok}  cached={cached}  failed={failed}")

    print(f"\nDone. ok={ok}  cached={cached}  failed={failed}")
    print(f"Videos in: {out_dir}")


if __name__ == "__main__":
    main()
