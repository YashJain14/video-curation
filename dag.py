"""
dag.py
------
Full pipeline orchestration using Prefect.

Stages run in order (each depends on the previous):
  1. ingest     — download Kinetics-400 clips
  2. embed      — CLIP embeddings (Ray parallel)
  3. dedup      — FAISS near-dedup
  4. score      — aesthetic scoring (Ray parallel)
  5. caption    — VLM captioning (Ray parallel)
  6. shard      — write WebDataset .tar shards
  7. manifest   — version the dataset

Why Prefect over Airflow here:
  - Zero-config local execution (no Postgres, no web server needed)
  - Same DAG concept as Airflow (flows, tasks, retries, schedules)
  - Easy to migrate to Prefect Cloud or swap for Airflow later
  - Conceptually identical to Airflow for interview purposes

Usage:
  # Run full pipeline
  python dag.py --csv data/kinetics400_val.csv --version v1

  # Run from a specific stage (skip completed earlier stages)
  python dag.py --csv data/kinetics400_val.csv --version v1 --from_stage score
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from datetime import timedelta


SCRATCH      = Path(os.environ.get("SCRATCH_DIR", "data"))
DATA_DIR     = SCRATCH
RAW_VIDEOS   = DATA_DIR / "raw_videos"
EMB_DIR      = DATA_DIR / "embeddings"
SHARDS_DIR   = DATA_DIR / "shards"
MANIFEST_DIR = DATA_DIR / "manifests"

STAGES = ["ingest", "embed", "dedup", "score", "caption", "shard", "manifest"]


def _run(cmd: list[str], stage: str):
    logger = get_run_logger()
    logger.info(f"[{stage}] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    logger.info(f"[{stage}] Done (exit={result.returncode})")


@task(name="ingest", retries=2, retry_delay_seconds=30,
      cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=24))
def task_ingest(split: str, limit: int, workers: int):
    _run([
        sys.executable, "ingest.py",
        "--split",   split,
        "--out_dir", str(RAW_VIDEOS),
        "--limit",   str(limit),
        "--workers", str(workers),
    ], "ingest")


@task(name="embed", retries=1)
def task_embed(frames_per_video: int, num_gpus: int):
    _run([
        sys.executable, "embed.py",
        "--video_dir",        str(RAW_VIDEOS),
        "--out_dir",          str(EMB_DIR),
        "--frames_per_video", str(frames_per_video),
        "--num_gpus",         str(num_gpus),
    ], "embed")


@task(name="dedup", retries=1)
def task_dedup(threshold: float):
    _run([
        sys.executable, "dedup.py",
        "--emb_dir",   str(EMB_DIR),
        "--threshold", str(threshold),
        "--out",       str(DATA_DIR / "dedup_results.json"),
    ], "dedup")


@task(name="score", retries=1)
def task_score(frames_per_video: int, num_gpus: int):
    _run([
        sys.executable, "score.py",
        "--video_dir",        str(RAW_VIDEOS),
        "--out",              str(DATA_DIR / "scores.json"),
        "--frames_per_video", str(frames_per_video),
        "--num_gpus",         str(num_gpus),
    ], "score")


@task(name="caption", retries=1)
def task_caption(frames_per_video: int, num_gpus: int):
    _run([
        sys.executable, "caption.py",
        "--video_dir",        str(RAW_VIDEOS),
        "--out",              str(DATA_DIR / "captions.json"),
        "--frames_per_video", str(frames_per_video),
        "--num_gpus",         str(num_gpus),
    ], "caption")


@task(name="shard", retries=1)
def task_shard(shard_size: int, min_score: float):
    _run([
        sys.executable, "shard.py",
        "--video_dir",  str(RAW_VIDEOS),
        "--captions",   str(DATA_DIR / "captions.json"),
        "--scores",     str(DATA_DIR / "scores.json"),
        "--dedup",      str(DATA_DIR / "dedup_results.json"),
        "--emb_dir",    str(EMB_DIR),
        "--out_dir",    str(SHARDS_DIR),
        "--shard_size", str(shard_size),
        "--min_score",  str(min_score),
    ], "shard")


@task(name="manifest")
def task_manifest(version: str, min_score: float, dedup_threshold: float):
    _run([
        sys.executable, "manifest.py", "create",
        "--version",         version,
        "--video_dir",       str(RAW_VIDEOS),
        "--dedup",           str(DATA_DIR / "dedup_results.json"),
        "--scores",          str(DATA_DIR / "scores.json"),
        "--shards",          str(SHARDS_DIR),
        "--out",             str(MANIFEST_DIR / f"{version}.json"),
        "--min_score",       str(min_score),
        "--dedup_threshold", str(dedup_threshold),
    ], "manifest")


@flow(name="video-curation-pipeline")
def curation_pipeline(
    split:             str   = "val",
    version:           str   = "v1",
    limit:             int   = 10,
    workers:           int   = 8,
    frames_per_video:  int   = 8,
    num_gpus:          int   = 1,
    dedup_threshold:   float = 0.95,
    min_score:         float = 4.5,
    shard_size:        int   = 200,
    from_stage:        str   = "ingest",
):
    """
    End-to-end video curation pipeline.
    Set from_stage to skip completed earlier stages.
    num_gpus is passed to Ray-based stages (embed, score, caption).
    """
    stage_idx = STAGES.index(from_stage)

    if stage_idx <= STAGES.index("ingest"):
        task_ingest(split, limit, workers)

    if stage_idx <= STAGES.index("embed"):
        task_embed(frames_per_video, num_gpus)

    if stage_idx <= STAGES.index("dedup"):
        task_dedup(dedup_threshold)

    if stage_idx <= STAGES.index("score"):
        task_score(frames_per_video, num_gpus)

    if stage_idx <= STAGES.index("caption"):
        task_caption(frames_per_video, num_gpus)

    if stage_idx <= STAGES.index("shard"):
        task_shard(shard_size, min_score)

    if stage_idx <= STAGES.index("manifest"):
        task_manifest(version, min_score, dedup_threshold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split",            default="val", choices=["train", "val", "test"])
    ap.add_argument("--version",          default="v1")
    ap.add_argument("--limit",            type=int,   default=500)
    ap.add_argument("--workers",          type=int,   default=8)
    ap.add_argument("--frames_per_video", type=int,   default=8)
    ap.add_argument("--num_gpus",         type=int,   default=1,
                    help="Number of GPUs to use for Ray-parallel stages")
    ap.add_argument("--dedup_threshold",  type=float, default=0.95)
    ap.add_argument("--min_score",        type=float, default=4.5)
    ap.add_argument("--shard_size",       type=int,   default=200)
    ap.add_argument("--from_stage",       default="ingest", choices=STAGES)
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
        shard_size=args.shard_size,
        from_stage=args.from_stage,
    )


if __name__ == "__main__":
    main()
