"""
shard.py
--------
Write curated videos into WebDataset .tar shards.

Each shard contains up to --shard_size samples. Each sample has:
  - <key>.mp4      : raw video bytes
  - <key>.txt      : VLM caption
  - <key>.json     : metadata (path, label, aesthetic_score, clip_embedding)

WebDataset shards are the standard format for training-scale dataloaders:
  - Sequential reads (no random seeks) → saturates disk/network bandwidth
  - Compatible with PyTorch DataLoader, HuggingFace datasets, NVIDIA DALI
  - Easy to stream from S3/GCS without full download

Usage:
  python shard.py \
    --video_dir   data/raw_videos \
    --captions    data/captions.json \
    --scores      data/scores.json \
    --dedup       data/dedup_results.json \
    --emb_dir     data/embeddings \
    --out_dir     data/shards \
    --shard_size  200 \
    --min_score   4.5
"""

import argparse
import io
import json
import tarfile
import time
from pathlib import Path

import numpy as np


def load_lookup(json_path: str, key: str = "path") -> dict:
    """Load a JSON list-of-dicts into a dict keyed by the `key` field."""
    with open(json_path) as f:
        items = json.load(f)
    if isinstance(items, dict):
        # dedup format: {"kept": [...], "removed": [...]}
        return items
    return {item[key]: item for item in items}


def load_embedding(emb_dir: Path, video_path: str) -> list | None:
    stem = Path(video_path).stem
    npz  = emb_dir / f"{stem}.npz"
    if not npz.exists():
        return None
    data = np.load(npz, allow_pickle=True)
    return data["mean_embedding"].tolist()


def write_shards(videos: list[str], captions: dict, scores: dict,
                 emb_dir: Path, out_dir: Path,
                 shard_size: int, min_score: float) -> list[str]:
    """
    Write videos to WebDataset .tar shards.
    Returns list of written shard paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = []
    current_tar = None
    current_path = None
    shard_idx  = 0
    sample_idx = 0

    for video_path in videos:
        # Filter by aesthetic score
        score_info = scores.get(video_path, {})
        mean_score = score_info.get("mean_score", 0.0)
        if mean_score < min_score:
            continue

        video_bytes = Path(video_path).read_bytes()
        caption     = captions.get(video_path, {}).get("caption", "")
        embedding   = load_embedding(emb_dir, video_path)
        label       = Path(video_path).parent.name

        key = f"{shard_idx:05d}_{sample_idx % shard_size:04d}"

        metadata = {
            "video_path":      video_path,
            "label":           label,
            "aesthetic_score": mean_score,
            "clip_embedding":  embedding,
        }

        # Open new shard if needed
        if current_tar is None or sample_idx % shard_size == 0 and sample_idx > 0:
            if current_tar is not None:
                current_tar.close()
                shard_paths.append(current_path)
            shard_idx   = sample_idx // shard_size
            current_path = str(out_dir / f"shard_{shard_idx:05d}.tar")
            current_tar  = tarfile.open(current_path, "w")

        def add_bytes(name, data: bytes):
            info        = tarfile.TarInfo(name=name)
            info.size   = len(data)
            current_tar.addfile(info, io.BytesIO(data))

        add_bytes(f"{key}.mp4",  video_bytes)
        add_bytes(f"{key}.txt",  caption.encode())
        add_bytes(f"{key}.json", json.dumps(metadata).encode())

        sample_idx += 1

    if current_tar is not None:
        current_tar.close()
        shard_paths.append(current_path)

    return shard_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir",  required=True)
    ap.add_argument("--captions",   required=True)
    ap.add_argument("--scores",     required=True)
    ap.add_argument("--dedup",      required=True)
    ap.add_argument("--emb_dir",    default="data/embeddings")
    ap.add_argument("--out_dir",    default="data/shards")
    ap.add_argument("--shard_size", type=int,   default=200)
    ap.add_argument("--min_score",  type=float, default=4.5)
    args = ap.parse_args()

    t0 = time.perf_counter()

    # Load lookups
    captions   = load_lookup(args.captions)   # {path: {caption, ...}}
    scores     = load_lookup(args.scores)      # {path: {mean_score, ...}}
    dedup      = load_lookup(args.dedup)       # {"kept": [...], "removed": [...]}

    kept_set = set(dedup.get("kept", []))
    all_videos = sorted(Path(args.video_dir).rglob("*.mp4"))

    # Only process kept (deduplicated) videos
    videos = [str(v) for v in all_videos if str(v) in kept_set] if kept_set else [str(v) for v in all_videos]
    print(f"Videos after dedup filter : {len(videos)} / {len(all_videos)}")

    shard_paths = write_shards(
        videos, captions, scores,
        Path(args.emb_dir), Path(args.out_dir),
        args.shard_size, args.min_score,
    )

    elapsed = time.perf_counter() - t0
    print(f"\nWrote {len(shard_paths)} shards → {args.out_dir}  ({elapsed:.1f}s)")
    for p in shard_paths:
        size_mb = Path(p).stat().st_size / 1e6
        print(f"  {Path(p).name}  {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
