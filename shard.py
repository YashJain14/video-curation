"""
shard.py
--------
Write curated videos into WebDataset .tar shards.

Each shard contains up to --shard_size samples. Each sample has:
  - <key>.mp4           : raw video bytes
  - <key>.txt           : VLM caption
  - <key>.json          : metadata (source, label, all quality scores,
                           clip_embedding, dataset composition tags)
  - <key>.latent.pt     : pre-encoded VAE latent [4, T, H//8, W//8] float16
                           (only written if --latent_dir is provided)

Dataset composition (v2):
  Each video is tagged with a source dataset name (--source, default "kinetics400").
  The shard manifest records per-source sample counts so training recipes can
  control dataset mix ratios (e.g. 70% Kinetics, 30% synthetic) by reading
  this metadata. Mixed datasets are produced by running shard.py separately
  per source then interleaving shards at training time via WebDataset's
  mix_shards / RandomMix — no re-sharding needed.

Quality gates applied before writing (v2):
  - aesthetic_score  ≥ min_score    (LAION aesthetic predictor)
  - motion_score     ≥ min_motion   (optical flow + SSIM composite)
  - caption quality  ≥ min_quality  (CLIP text-image alignment composite)
  All three signals are stored in each sample's .json for downstream analysis.

WebDataset shards are the standard format for training-scale dataloaders:
  - Sequential reads (no random seeks) → saturates disk/network bandwidth
  - Compatible with PyTorch DataLoader, HuggingFace datasets, NVIDIA DALI
  - Easy to stream from S3/GCS without full download

Usage:
  python shard.py \\
    --video_dir     data/raw_videos \\
    --captions      data/captions.json \\
    --scores        data/scores.json \\
    --dedup         data/dedup_results.json \\
    --emb_dir       data/embeddings \\
    --motion        data/motion_scores.json \\
    --caption_quality data/caption_quality.json \\
    --latent_dir    data/latents \\
    --out_dir       data/shards \\
    --source        kinetics400 \\
    --shard_size    200 \\
    --min_score     4.5 \\
    --min_motion    3.0 \\
    --min_quality   0.3
"""

import argparse
import io
import json
import tarfile
import time
from pathlib import Path

import numpy as np
import wandb


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
    data = np.load(npz)
    return data["mean_embedding"].tolist()


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes):
    info      = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def write_shards(
    videos: list[str],
    captions: dict,
    scores: dict,
    motion: dict,
    caption_quality: dict,
    emb_dir: Path,
    latent_dir: Path | None,
    out_dir: Path,
    shard_size: int,
    min_score: float,
    min_motion: float,
    min_quality: float,
    source: str,
) -> tuple[list[str], dict]:
    """
    Write videos to WebDataset .tar shards.
    Returns (list of written shard paths, composition summary dict).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_paths    = []
    current_tar    = None
    current_path   = None
    shard_idx      = 0
    sample_idx     = 0
    skipped_score  = skipped_motion = skipped_quality = skipped_caption = 0

    for video_path in videos:
        # --- quality gates ---
        score_info = scores.get(video_path, {})
        mean_score = score_info.get("mean_score", 0.0)
        if mean_score < min_score:
            skipped_score += 1
            continue

        motion_info  = motion.get(video_path, {})
        motion_score = motion_info.get("motion_score", 0.0)
        if motion and motion_score < min_motion:
            skipped_motion += 1
            continue

        cq_info      = caption_quality.get(video_path, {})
        quality_score = cq_info.get("quality_score", 1.0)   # default 1.0 if no CQ run
        if caption_quality and quality_score < min_quality:
            skipped_quality += 1
            continue

        caption = captions.get(video_path, {}).get("caption", "")
        if not caption:
            skipped_caption += 1
            continue

        # --- build shard entry ---
        video_bytes = Path(video_path).read_bytes()
        embedding   = load_embedding(emb_dir, video_path)
        label       = Path(video_path).parent.name
        key         = f"{shard_idx:05d}_{sample_idx % shard_size:04d}"

        metadata = {
            "video_path":       video_path,
            "source":           source,
            "label":            label,
            "aesthetic_score":  mean_score,
            "motion_score":     motion_score,
            "caption_quality":  quality_score,
            "clip_alignment":   cq_info.get("clip_alignment", None),
            "caption_length":   cq_info.get("caption_length", None),
            "clip_embedding":   embedding,
        }

        # Open new shard tar when needed
        if current_tar is None or (sample_idx % shard_size == 0 and sample_idx > 0):
            if current_tar is not None:
                current_tar.close()
                shard_paths.append(current_path)
            shard_idx    = sample_idx // shard_size
            current_path = str(out_dir / f"shard_{shard_idx:05d}.tar")
            current_tar  = tarfile.open(current_path, "w")

        _add_bytes(current_tar, f"{key}.mp4",  video_bytes)
        _add_bytes(current_tar, f"{key}.txt",  caption.encode())
        _add_bytes(current_tar, f"{key}.json", json.dumps(metadata).encode())

        # Embed pre-encoded latent if available
        if latent_dir is not None:
            latent_path = latent_dir / (Path(video_path).stem + ".pt")
            if latent_path.exists():
                _add_bytes(current_tar, f"{key}.latent.pt", latent_path.read_bytes())

        sample_idx += 1

    if current_tar is not None:
        current_tar.close()
        shard_paths.append(current_path)

    composition = {
        "source":          source,
        "total_written":   sample_idx,
        "skipped_score":   skipped_score,
        "skipped_motion":  skipped_motion,
        "skipped_quality": skipped_quality,
        "skipped_caption": skipped_caption,
    }
    return shard_paths, composition


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir",        required=True)
    ap.add_argument("--captions",         required=True)
    ap.add_argument("--scores",           required=True)
    ap.add_argument("--dedup",            required=True)
    ap.add_argument("--emb_dir",          default="data/embeddings")
    ap.add_argument("--motion",           default=None,
                    help="motion_scores.json from motion.py (optional)")
    ap.add_argument("--caption_quality",  default=None,
                    help="caption_quality.json from caption_quality.py (optional)")
    ap.add_argument("--latent_dir",       default=None,
                    help="Directory of .pt latent files from encode_latents.py (optional)")
    ap.add_argument("--out_dir",          default="data/shards")
    ap.add_argument("--source",           default="kinetics400",
                    help="Dataset source tag embedded in every sample's metadata. "
                         "Used for composition tracking in mixed-dataset training.")
    ap.add_argument("--shard_size",       type=int,   default=200)
    ap.add_argument("--min_score",        type=float, default=4.5)
    ap.add_argument("--min_motion",       type=float, default=3.0)
    ap.add_argument("--min_quality",      type=float, default=0.3)
    args = ap.parse_args()

    t0 = time.perf_counter()

    wandb.init(project="video-curation", entity="rlx-labs",
               name="shard-stage", resume="allow",
               config={"shard_size":   args.shard_size,
                       "min_score":    args.min_score,
                       "min_motion":   args.min_motion,
                       "min_quality":  args.min_quality,
                       "source":       args.source,
                       "has_latents":  args.latent_dir is not None,
                       "out_dir":      args.out_dir})

    captions        = load_lookup(args.captions)
    scores          = load_lookup(args.scores)
    dedup           = load_lookup(args.dedup)
    motion          = load_lookup(args.motion)  if args.motion          else {}
    caption_quality = load_lookup(args.caption_quality) if args.caption_quality else {}

    kept_set   = set(dedup.get("kept", []))
    all_videos = sorted(Path(args.video_dir).rglob("*.mp4"))
    videos     = ([str(v) for v in all_videos if str(v) in kept_set]
                  if kept_set else [str(v) for v in all_videos])
    print(f"Videos after dedup filter : {len(videos)} / {len(all_videos)}")

    latent_dir = Path(args.latent_dir) if args.latent_dir else None

    shard_paths, composition = write_shards(
        videos, captions, scores, motion, caption_quality,
        Path(args.emb_dir), latent_dir, Path(args.out_dir),
        args.shard_size, args.min_score, args.min_motion, args.min_quality,
        args.source,
    )

    elapsed     = time.perf_counter() - t0
    total_bytes = 0
    print(f"\nWrote {len(shard_paths)} shards → {args.out_dir}  ({elapsed:.1f}s)")
    for p in shard_paths:
        size_b       = Path(p).stat().st_size
        total_bytes += size_b
        print(f"  {Path(p).name}  {size_b/1e6:.1f} MB")

    print(f"\nDataset composition [{args.source}]:")
    print(f"  Written         : {composition['total_written']}")
    print(f"  Skipped (score) : {composition['skipped_score']}")
    print(f"  Skipped (motion): {composition['skipped_motion']}")
    print(f"  Skipped (quality): {composition['skipped_quality']}")
    print(f"  Skipped (no caption): {composition['skipped_caption']}")

    wandb.log({
        "shard/n_shards":            len(shard_paths),
        "shard/n_written":           composition["total_written"],
        "shard/skipped_score":       composition["skipped_score"],
        "shard/skipped_motion":      composition["skipped_motion"],
        "shard/skipped_quality":     composition["skipped_quality"],
        "shard/skipped_caption":     composition["skipped_caption"],
        "shard/total_bytes":         total_bytes,
        "shard/total_mb":            total_bytes / 1e6,
        "shard/elapsed_s":           elapsed,
        "shard/has_latents":         latent_dir is not None,
        "shard/source":              args.source,
    })
    wandb.finish()


if __name__ == "__main__":
    main()
