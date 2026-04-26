"""
dag.py
------
Full pipeline orchestration using Prefect.

v1 stages (original):
  1. ingest     — download Kinetics-400 clips from S3
  2. embed      — CLIP ViT-B/32 embeddings (Ray, 0.25 GPU/task)
  3. dedup      — MD5 exact + FAISS near-dedup
  4. score      — LAION aesthetic scoring (Ray, 0.25 GPU/task)
  5. caption    — Qwen3-VL-8B captioning (Ray, 1.0 GPU/task)
  6. shard      — write WebDataset .tar shards
  7. manifest   — versioned dataset manifest

v2 additions (research-grade curation):
  motion          — optical flow + SSIM motion quality (CPU Ray, after score)
  caption_quality — CLIP text-image alignment scoring (Ray, after caption)
  encode_latents  — pre-encode VAE latents for training efficiency (Ray, after caption_quality)

New stages slot into the DAG between score→shard and caption→shard:
  score → motion → (filtered) → caption → caption_quality → encode_latents → shard

Why Prefect over Airflow here:
  - Zero-config local execution (no Postgres, no web server needed)
  - Same DAG concept as Airflow (flows, tasks, retries, schedules)
  - Easy to migrate to Prefect Cloud or swap for Airflow later
  - Conceptually identical to Airflow for interview purposes

Usage:
  # Run full pipeline (v1 stages only)
  python dag.py --version v1 --from_stage ingest --num_gpus 4

  # Run full pipeline including v2 quality stages
  python dag.py --version v2 --from_stage ingest --num_gpus 4 --enable_v2

  # Resume from a specific stage
  python dag.py --version v2 --from_stage caption_quality --num_gpus 4 --enable_v2
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import wandb
from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from datetime import timedelta


def _paths():
    scratch = Path(os.environ.get("SCRATCH_DIR") or os.path.expanduser("~/scratch/video-curation"))
    return {
        "data":     scratch,
        "videos":   scratch / "raw_videos",
        "emb":      scratch / "embeddings",
        "shards":   scratch / "shards",
        "manifest": scratch / "manifests",
    }

STAGES = ["ingest", "embed", "dedup", "score", "motion", "caption",
          "caption_quality", "encode_latents", "shard", "manifest"]


def _run(cmd: list[str], stage: str):
    logger = get_run_logger()
    logger.info(f"[{stage}] Running: {' '.join(cmd)}")
    t0 = time.perf_counter()
    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.perf_counter() - t0
        logger.info(f"[{stage}] Done (exit={result.returncode})")
        wandb.log({f"{stage}/status": 1, f"{stage}/duration_s": elapsed})
    except subprocess.CalledProcessError as e:
        elapsed = time.perf_counter() - t0
        logger.error(f"[{stage}] Failed (exit={e.returncode})")
        wandb.log({f"{stage}/status": 0, f"{stage}/duration_s": elapsed})
        raise


@task(name="ingest", retries=2, retry_delay_seconds=30,
      cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=24))
def task_ingest(split: str, limit: int, workers: int):
    p = _paths()
    _run([
        sys.executable, "ingest.py",
        "--split",   split,
        "--out_dir", str(p["videos"]),
        "--limit",   str(limit),
        "--workers", str(workers),
    ], "ingest")


@task(name="embed", retries=1)
def task_embed(frames_per_video: int, num_gpus: int):
    p = _paths()
    _run([
        sys.executable, "embed.py",
        "--video_dir",        str(p["videos"]),
        "--out_dir",          str(p["emb"]),
        "--frames_per_video", str(frames_per_video),
        "--num_gpus",         str(num_gpus),
    ], "embed")


@task(name="dedup", retries=1)
def task_dedup(threshold: float):
    p = _paths()
    _run([
        sys.executable, "dedup.py",
        "--video_dir",  str(p["videos"]),
        "--emb_dir",    str(p["emb"]),
        "--threshold",  str(threshold),
        "--out",        str(p["data"] / "dedup_results.json"),
        "--index_path", str(p["data"] / "faiss.index"),
        "--md5_cache",  str(p["data"] / "md5_cache.json"),
    ], "dedup")


@task(name="score", retries=1)
def task_score(frames_per_video: int, num_gpus: int):
    p = _paths()
    _run([
        sys.executable, "score.py",
        "--video_dir",        str(p["videos"]),
        "--out",              str(p["data"] / "scores.json"),
        "--frames_per_video", str(frames_per_video),
        "--num_gpus",         str(num_gpus),
    ], "score")


@task(name="motion", retries=1)
def task_motion(num_workers: int):
    p = _paths()
    _run([
        sys.executable, "motion.py",
        "--video_dir",   str(p["videos"]),
        "--out",         str(p["data"] / "motion_scores.json"),
        "--num_workers", str(num_workers),
    ], "motion")


@task(name="caption", retries=1)
def task_caption(frames_per_video: int, num_gpus: int,
                 min_score: float, min_motion: float, enable_v2: bool):
    p = _paths()
    cmd = [
        sys.executable, "caption.py",
        "--video_dir",        str(p["videos"]),
        "--out",              str(p["data"] / "captions.json"),
        "--frames_per_video", str(frames_per_video),
        "--num_gpus",         str(num_gpus),
        "--scores",           str(p["data"] / "scores.json"),
        "--min_score",        str(min_score),
    ]
    # In v2, motion scores are available before captioning — filter here to
    # avoid paying Qwen3-VL compute on clips that will fail motion gate at shard.
    if enable_v2:
        motion_path = p["data"] / "motion_scores.json"
        if motion_path.exists():
            cmd += ["--motion",     str(motion_path),
                    "--min_motion", str(min_motion)]
    _run(cmd, "caption")


@task(name="caption_quality", retries=1)
def task_caption_quality(num_gpus: int):
    p = _paths()
    _run([
        sys.executable, "caption_quality.py",
        "--captions", str(p["data"] / "captions.json"),
        "--emb_dir",  str(p["emb"]),
        "--out",      str(p["data"] / "caption_quality.json"),
        "--num_gpus", str(num_gpus),
    ], "caption_quality")


@task(name="encode_latents", retries=1)
def task_encode_latents(num_gpus: int, num_frames: int, resolution: int,
                         min_score: float, min_motion: float, min_quality: float):
    p = _paths()
    _run([
        sys.executable, "encode_latents.py",
        "--video_dir",        str(p["videos"]),
        "--out_dir",          str(p["data"] / "latents"),
        "--scores",           str(p["data"] / "scores.json"),
        "--motion",           str(p["data"] / "motion_scores.json"),
        "--captions_quality", str(p["data"] / "caption_quality.json"),
        "--min_score",        str(min_score),
        "--min_motion",       str(min_motion),
        "--min_quality",      str(min_quality),
        "--num_gpus",         str(num_gpus),
        "--num_frames",       str(num_frames),
        "--resolution",       str(resolution),
    ], "encode_latents")


@task(name="shard", retries=1)
def task_shard(shard_size: int, min_score: float, min_motion: float,
               min_quality: float, source: str, enable_v2: bool):
    p = _paths()
    cmd = [
        sys.executable, "shard.py",
        "--video_dir",  str(p["videos"]),
        "--captions",   str(p["data"] / "captions.json"),
        "--scores",     str(p["data"] / "scores.json"),
        "--dedup",      str(p["data"] / "dedup_results.json"),
        "--emb_dir",    str(p["emb"]),
        "--out_dir",    str(p["shards"]),
        "--shard_size", str(shard_size),
        "--min_score",  str(min_score),
        "--source",     source,
    ]
    if enable_v2:
        motion_path = p["data"] / "motion_scores.json"
        cq_path     = p["data"] / "caption_quality.json"
        latent_dir  = p["data"] / "latents"
        if motion_path.exists():
            cmd += ["--motion",          str(motion_path),
                    "--min_motion",      str(min_motion)]
        if cq_path.exists():
            cmd += ["--caption_quality", str(cq_path),
                    "--min_quality",     str(min_quality)]
        if latent_dir.exists():
            cmd += ["--latent_dir",      str(latent_dir)]
    _run(cmd, "shard")


@task(name="manifest")
def task_manifest(version: str, min_score: float, dedup_threshold: float,
                  min_motion: float = 5.0, min_quality: float = 0.65,
                  enable_v2: bool = True):
    p = _paths()
    cmd = [
        sys.executable, "manifest.py", "create",
        "--version",         version,
        "--video_dir",       str(p["videos"]),
        "--dedup",           str(p["data"] / "dedup_results.json"),
        "--scores",          str(p["data"] / "scores.json"),
        "--shards",          str(p["shards"]),
        "--out",             str(p["manifest"] / f"{version}.json"),
        "--min_score",       str(min_score),
        "--dedup_threshold", str(dedup_threshold),
    ]
    if enable_v2:
        motion_path = p["data"] / "motion_scores.json"
        cq_path     = p["data"] / "caption_quality.json"
        if motion_path.exists():
            cmd += ["--motion",    str(motion_path), "--min_motion",  str(min_motion)]
        if cq_path.exists():
            cmd += ["--caption_quality", str(cq_path), "--min_quality", str(min_quality)]
    _run(cmd, "manifest")


@flow(name="video-curation-pipeline")
def curation_pipeline(
    split:             str   = "val",
    version:           str   = "v2",
    limit:             int   = 10,
    workers:           int   = 8,
    frames_per_video:  int   = 8,
    num_gpus:          int   = 1,
    dedup_threshold:   float = 0.95,
    min_score:         float = 4.5,
    min_motion:        float = 5.0,
    min_quality:       float = 0.65,
    shard_size:        int   = 200,
    source:            str   = "kinetics400",
    latent_frames:     int   = 16,
    latent_resolution: int   = 256,
    from_stage:        str   = "ingest",
    enable_v2:         bool  = True,
):
    """
    End-to-end video curation pipeline (v2).

    enable_v2=True adds three research-grade curation stages:
      motion          — optical flow + SSIM motion quality signal
      caption_quality — CLIP text-image alignment score
      encode_latents  — pre-encode VAE latents for training efficiency

    Set from_stage to skip completed earlier stages.
    num_gpus is passed to all Ray-based GPU stages.
    """
    wandb.init(
        project="video-curation",
        entity="rlx-labs",
        name=f"{version}-{from_stage}",
        config={
            "split": split, "version": version, "limit": limit,
            "workers": workers, "frames_per_video": frames_per_video,
            "num_gpus": num_gpus, "dedup_threshold": dedup_threshold,
            "min_score": min_score, "min_motion": min_motion,
            "min_quality": min_quality, "shard_size": shard_size,
            "source": source, "enable_v2": enable_v2,
            "latent_frames": latent_frames, "latent_resolution": latent_resolution,
            "from_stage": from_stage,
        },
    )

    stage_idx = STAGES.index(from_stage)

    if stage_idx <= STAGES.index("ingest"):
        task_ingest(split, limit, workers)

    if stage_idx <= STAGES.index("embed"):
        task_embed(frames_per_video, num_gpus)

    if stage_idx <= STAGES.index("dedup"):
        task_dedup(dedup_threshold)

    if stage_idx <= STAGES.index("score"):
        task_score(frames_per_video, num_gpus)

    if enable_v2 and stage_idx <= STAGES.index("motion"):
        # CPU-only: use available CPU cores; 4 workers per GPU is a reasonable proxy
        task_motion(num_workers=num_gpus * 4)

    if stage_idx <= STAGES.index("caption"):
        task_caption(frames_per_video, num_gpus, min_score, min_motion, enable_v2)

    if enable_v2 and stage_idx <= STAGES.index("caption_quality"):
        task_caption_quality(num_gpus)

    if enable_v2 and stage_idx <= STAGES.index("encode_latents"):
        task_encode_latents(num_gpus, latent_frames, latent_resolution,
                            min_score, min_motion, min_quality)

    if stage_idx <= STAGES.index("shard"):
        task_shard(shard_size, min_score, min_motion, min_quality, source, enable_v2)

    if stage_idx <= STAGES.index("manifest"):
        task_manifest(version, min_score, dedup_threshold,
                      min_motion=min_motion, min_quality=min_quality,
                      enable_v2=enable_v2)

    wandb.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split",             default="val", choices=["train", "val", "test"])
    ap.add_argument("--version",           default="v2")
    ap.add_argument("--limit",             type=int,   default=500)
    ap.add_argument("--workers",           type=int,   default=8)
    ap.add_argument("--frames_per_video",  type=int,   default=8)
    ap.add_argument("--num_gpus",          type=int,   default=1,
                    help="Number of GPUs for Ray-parallel stages")
    ap.add_argument("--dedup_threshold",   type=float, default=0.95)
    ap.add_argument("--min_score",         type=float, default=4.5)
    ap.add_argument("--min_motion",        type=float, default=5.0,
                    help="Motion quality threshold (v2 only); p25 of Kinetics distribution")
    ap.add_argument("--min_quality",       type=float, default=0.65,
                    help="Caption quality threshold (v2 only); ~p50 of quality distribution")
    ap.add_argument("--shard_size",        type=int,   default=200)
    ap.add_argument("--source",            default="kinetics400",
                    help="Dataset source tag for composition tracking")
    ap.add_argument("--latent_frames",     type=int,   default=16,
                    help="Frames per video for VAE latent encoding (v2 only)")
    ap.add_argument("--latent_resolution", type=int,   default=256,
                    help="Spatial resolution before VAE encoding (v2 only)")
    ap.add_argument("--from_stage",        default="ingest", choices=STAGES)
    ap.add_argument("--enable_v2",         action="store_true", default=True,
                    help="Enable v2 stages: motion, caption_quality, encode_latents")
    ap.add_argument("--no_v2",             action="store_true",
                    help="Disable v2 stages (run v1-only pipeline)")
    args = ap.parse_args()

    curation_pipeline(
        split=args.split,
        version=args.version,
        limit=args.limit,
        workers=args.workers,
        frames_per_video=args.frames_per_video,
        num_gpus=args.num_gpus,
        dedup_threshold=args.dedup_threshold,
        min_score=args.min_score,
        min_motion=args.min_motion,
        min_quality=args.min_quality,
        shard_size=args.shard_size,
        source=args.source,
        latent_frames=args.latent_frames,
        latent_resolution=args.latent_resolution,
        from_stage=args.from_stage,
        enable_v2=not args.no_v2,
    )


if __name__ == "__main__":
    main()
