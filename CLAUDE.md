# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

End-to-end video curation pipeline: downloads Kinetics-400 clips from S3, runs CLIP embeddings, two-stage deduplication, aesthetic scoring, VLM captioning, and writes training-ready WebDataset shards with versioned manifests.

**Hardware target:** NVIDIA A100-SXM4-40GB · CUDA 12.4 · PyTorch 2.6  
**Cluster:** NSCC HPC · PBS scheduler · 4× A100 GPUs per node

## v2 Changes (Research-Grade Curation Branch)

Three new stages added to close the gap between raw curation and what video diffusion model training actually needs:

| Stage | Script | Purpose |
|-------|--------|---------|
| `motion` | `motion.py` | Optical flow magnitude + SSIM between frames → `motion_score [0–10]`. Catches shaky footage and scene cuts that aesthetic score misses. |
| `caption_quality` | `caption_quality.py` | CLIP text-image cosine alignment + caption length + specificity → `quality_score [0–1]`. Ensures VLM captions are semantically accurate (captions are the text conditioning signal for the diffusion model). |
| `encode_latents` | `encode_latents.py` | Pre-encode video frames through SD-VAE into `[4, T, H//8, W//8]` float16 tensors. Eliminates per-step VAE encoding cost during training (~15–30% training GPU savings). |

`shard.py` now gates on all three signals, tags each sample with `--source`, and embeds pre-encoded latents (`<key>.latent.pt`) in shards when `--latent_dir` is provided.

`dag.py` is fully backward-compatible: `--no_v2` runs the original 7-stage pipeline. Default is v2 (`--enable_v2`).

New `$SCRATCH_DIR` outputs:
```
motion_cache/<stem>.motion.json
motion_scores.json
caption_quality_cache/<stem>.cq.json
caption_quality.json
latents/<stem>.pt                  # float16 [4, T, H//8, W//8]
```

## Running the Pipeline

### Setup (login node, once)

```bash
bash setup_env.sh
python prefetch_models.py   # downloads CLIP ViT-B/32, ViT-L/14, LAION aesthetic predictor, Qwen3-VL-8B (~16 GB)
git clone https://github.com/YashJain14/torchscope.git ~/torchscope
```

Compute nodes have no internet (`HF_HUB_OFFLINE=1`). All HuggingFace models must be pre-fetched from the login node.

### Full pipeline

```bash
qsub run_curation.pbs          # PBS job: all 7 stages, logs to curation_output.log
```

### Resume from a stage

```bash
# v2 pipeline (default — includes motion, caption_quality, encode_latents)
python dag.py --version v2 --limit 10000 --from_stage caption_quality --num_gpus 4
python dag.py --version v2 --limit 10000 --from_stage encode_latents --num_gpus 4
python dag.py --version v2 --limit 10000 --from_stage shard --num_gpus 4

# v1 pipeline (skip v2 stages)
python dag.py --version v1 --limit 10000 --from_stage caption --num_gpus 4 --no_v2
```

Valid `--from_stage` values: `ingest`, `embed`, `dedup`, `score`, `motion`, `caption`, `caption_quality`, `encode_latents`, `shard`, `manifest`

### Run individual stages

```bash
export SCRATCH_DIR="$HOME/scratch/video-curation"

python ingest.py --split val --out_dir $SCRATCH_DIR/raw_videos --limit 10
python embed.py --video_dir $SCRATCH_DIR/raw_videos --out_dir $SCRATCH_DIR/embeddings --num_gpus 4
python dedup.py --video_dir $SCRATCH_DIR/raw_videos --emb_dir $SCRATCH_DIR/embeddings --threshold 0.95 --index_path $SCRATCH_DIR/faiss.index
python score.py --video_dir $SCRATCH_DIR/raw_videos --out $SCRATCH_DIR/scores.json --num_gpus 4

# v2 stages
python motion.py --video_dir $SCRATCH_DIR/raw_videos --out $SCRATCH_DIR/motion_scores.json --num_workers 16
python caption.py --video_dir $SCRATCH_DIR/raw_videos --out $SCRATCH_DIR/captions.json --scores $SCRATCH_DIR/scores.json --min_score 4.5 --num_gpus 4
python caption_quality.py --captions $SCRATCH_DIR/captions.json --emb_dir $SCRATCH_DIR/embeddings --out $SCRATCH_DIR/caption_quality.json --num_gpus 1
python encode_latents.py --video_dir $SCRATCH_DIR/raw_videos --out_dir $SCRATCH_DIR/latents --scores $SCRATCH_DIR/scores.json --motion $SCRATCH_DIR/motion_scores.json --captions_quality $SCRATCH_DIR/caption_quality.json --num_gpus 4

# shard with all v2 signals
python shard.py --video_dir $SCRATCH_DIR/raw_videos --captions $SCRATCH_DIR/captions.json \
    --scores $SCRATCH_DIR/scores.json --dedup $SCRATCH_DIR/dedup_results.json \
    --emb_dir $SCRATCH_DIR/embeddings --motion $SCRATCH_DIR/motion_scores.json \
    --caption_quality $SCRATCH_DIR/caption_quality.json --latent_dir $SCRATCH_DIR/latents \
    --out_dir $SCRATCH_DIR/shards --source kinetics400 --min_score 4.5 --min_motion 3.0 --min_quality 0.3

python manifest.py create --version v2 --video_dir $SCRATCH_DIR/raw_videos --dedup $SCRATCH_DIR/dedup_results.json --scores $SCRATCH_DIR/scores.json --shards $SCRATCH_DIR/shards --out $SCRATCH_DIR/manifests/v2.json
python manifest.py diff --a $SCRATCH_DIR/manifests/v1.json --b $SCRATCH_DIR/manifests/v2.json
```

### GPU profiling

```bash
qsub run_profile.pbs
# HTML reports appear in $SCRATCH_DIR/reports/
```

## Architecture

### Pipeline DAG (`dag.py`)

Orchestrated via **Prefect** (`@flow` / `@task`). Each stage is a Prefect task that shells out to the corresponding script. `dag.py` reads `$SCRATCH_DIR` (defaults to `~/scratch/video-curation`) for all I/O paths.

```
ingest → embed → dedup → score → motion* → caption → caption_quality* → encode_latents* → shard → manifest
                                  (* v2 only; skipped with --no_v2)
```

All stages log timing to **wandb** (project `video-curation`, entity `rlx-labs`). Do not add hardcoded `id=` to `wandb.init()` calls — it breaks retries.

### Ray concurrency model

| Stage           | GPU fraction | Concurrent actors (4 GPUs) |
|-----------------|-------------|---------------------------|
| embed           | 0.25        | 16                        |
| score           | 0.25        | 16                        |
| caption         | 1.0         | 4                         |
| motion          | CPU only    | 16 workers (2 CPU each)   |
| caption_quality | 0.5         | 8                         |
| encode_latents  | 0.5         | 8                         |

### Decode path (`decode.py`)

Two decode functions with different use cases:

- **`decode_video_actor()`** — CPU decode via PyAV. Used by all Ray actors (`embed.py`, `score.py`, `caption.py`). Must stay CPU — fractional-GPU actors share one physical GPU, and PyNvVideoCodec creates a CUDA context per decoder instance, causing `CUDA_ERROR_CONTEXT_IS_DESTROYED` when contexts race.
- **`decode_video()`** — GPU decode via `ThreadedDecoder` (PyNvVideoCodec). Used only by `profile_run.py` where one process owns the GPU.

Never switch Ray actors back to GPU decode.

### Models

| Model | Repo | Used in |
|-------|------|---------|
| CLIP ViT-B/32 | `openai/clip-vit-base-patch32` | `embed.py`, `filter.py`, `caption_quality.py` |
| CLIP ViT-L/14 | `openai/clip-vit-large-patch14` | `score.py` |
| LAION aesthetic MLP | `camenduru/improved-aesthetic-predictor` (`sac+logos+ava1-l14-linearMSE.pth`) | `score.py` |
| Qwen3-VL-8B | `Qwen/Qwen3-VL-8B-Instruct` | `caption.py` |
| SD-VAE | `stabilityai/sd-vae-ft-mse` | `encode_latents.py` (v2) |

The aesthetic MLP was trained on ViT-L/14 embeddings specifically — cannot swap the backbone.

### Qwen3-VL API (caption.py)

Use `Qwen3VLForConditionalGeneration` (not the Qwen2.5 class). Load with `dtype="auto"`. The `qwen-vl-utils==0.0.14` library returns `fps` as a list instead of a scalar when `return_video_kwargs=True` — the fix is applied in `caption.py`:

```python
if "fps" in video_kwargs and isinstance(video_kwargs["fps"], list):
    video_kwargs["fps"] = video_kwargs["fps"][0] if video_kwargs["fps"] else 1.0
```

Do not pass `return_video_metadata=True`.

### Deduplication (`dedup.py`)

Two-stage: MD5 exact dedup (O(N)) → FAISS `IndexFlatIP` near-dedup (cosine similarity > 0.95, with Union-Find clustering). The FAISS index is saved to `faiss.index` and reloaded on reruns. For >500k videos, switch to `IndexIVFFlat(nlist=1024)`.

### Output directory layout

All output lives under `$SCRATCH_DIR` (`~/scratch/video-curation` by default):

```
raw_videos/<label>/<yt_id>_<start>_<end>.mp4
embeddings/<video_stem>.npz           # mean_embedding [512], frame_embeddings [8,512]
faiss.index
dedup_results.json
score_cache/<video_stem>.score.json
scores.json
caption_cache/<video_stem>.caption.json
captions.json
shards/shard_XXXXX.tar                # WebDataset: .mp4 + .txt + .json per sample
manifests/v1.json
reports/embed_cluster_profile.html
```

### Caching

Every GPU-intensive stage (embed, score, caption) skips already-processed videos by checking for per-video cache files (`.npz`, `.score.json`, `.caption.json`). Reruns are fast.

## Key Constraints

- **numpy < 2**: `faiss-gpu` is compiled against the NumPy 1.x ABI. Upgrading breaks `import faiss`.
- **opencv-python-headless < 4.11**: versions ≥ 4.11 are built against NumPy 2 ABI; incompatible with the numpy pin.
- **`torchscope`** is not pip-installed; it is used via local checkout. Pass `--torchscope ~/torchscope` to `profile_run.py`.
- **Aesthetic predictor** weights must be fetched from `camenduru/improved-aesthetic-predictor`. The `shunk031/` and `christophschuhmann/` repos don't work.
