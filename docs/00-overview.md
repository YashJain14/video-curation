# Pipeline Overview

End-to-end video curation pipeline: downloads Kinetics-400 clips from S3, runs CLIP embeddings, two-stage deduplication, aesthetic scoring, motion quality scoring, VLM captioning, caption quality analysis, VAE latent pre-encoding, and writes training-ready WebDataset shards with versioned manifests.

**Hardware target:** NVIDIA A100-SXM4-40GB · CUDA 12.4 · PyTorch 2.6  
**Cluster:** NSCC HPC · PBS scheduler · 4× A100 GPUs per node

---

## Pipeline DAG

```
Kinetics-400 S3 (CVDF)
       │
       ▼
[Stage 1:  ingest.py]           Download tar.gz from S3, extract MP4s into <label>/ dirs
       │
       ▼
[Stage 2:  embed.py]            CLIP ViT-B/32 embeddings         (Ray, 0.25 GPU/task → 16 concurrent)
       │
       ▼
[Stage 3:  dedup.py]            MD5 exact + FAISS near-dedup      (MD5 cache + persistent FAISS index)
       │
       ▼
[Stage 4:  score.py]            LAION aesthetic scorer            (Ray, 0.25 GPU/task → 16 concurrent)
       │
       ▼
[Stage 5:  motion.py]           Optical flow + SSIM quality       (CPU Ray, 16 workers, cached)
       │                         runs on all videos — filters before VLM
       ▼
[Stage 6:  caption.py]          Qwen3-VL-8B captioning            (Ray, 1.0 GPU/task → 4 concurrent)
       │                         only videos passing aesthetic + motion gates
       ▼
[Stage 7:  caption_quality.py]  CLIP text-image alignment         (Ray, 0.5 GPU/task → 8 concurrent)
       │                         verifies VLM output is semantically accurate
       ▼
[Stage 8:  encode_latents.py]   SD-VAE latent pre-encoding        (Ray, 0.5 GPU/task → 8 concurrent)
       │                         only final-passing videos; ~15–30% training GPU savings
       ▼
[Stage 9:  shard.py]            WebDataset .tar shards            (video + caption + scores + latent)
       │                         gates on all three quality signals
       ▼
[Stage 10: manifest.py]         Versioned dataset manifest        (full funnel stats + wandb)
```

Orchestrated by `dag.py` (Prefect `@flow` / `@task`). Each stage shells out to the corresponding script. Resume from any stage with `--from_stage`. Skip stages 5/7/8 with `--no_v2`.

---

## Run Results — 10,000 Kinetics-400 val clips, 4× A100-SXM4-40GB

**Quality funnel (v2, from `curation_output.log`):**
```
10,000 ingested
  └─ 9,998 embedded  (2 corrupt videos — IndexError: tuple index out of range)
      └─ 9,944 after dedup  (54 near-duplicates removed at cosine sim > 0.95)
          └─ 2,993 passing aesthetic ≥ 4.5  (29.9%)
              └─ 2,396 passing motion ≥ 5.0  (80.1% of aesthetic-passing)
                  └─ 2,396 captioned  (597 fewer VLM calls vs aesthetic-only gate)
                      └─ 2,161 passing caption quality ≥ 0.65  (90.2%)
                          └─ 2,161 latents encoded  [4, 16, 32, 32] float16
                              └─ 2,155 written to 11 shards  (4.1 GB)
```

**Runtime (wall clock, 4× A100):**

| Stage | Time | Notes |
|-------|------|-------|
| ingest | — | S3 download, one-time |
| embed | 2:03 | 9,998 cached; fast path through cache check |
| dedup | 10:15 | MD5 cold (no cache yet); FAISS cold build |
| score | ~2 min | Cached from previous run |
| motion | ~4 min | CPU-only, 16 workers, per-video cached |
| caption | 4:30 | 2,396 videos × 4 GPUs |
| caption_quality | 2:02 | 2,396 videos, 8 GPU actors |
| encode_latents | 41:42 | 2,161 videos, batched VAE encode |
| shard | 14s | Pure I/O |
| manifest | ~12s | Reads tars, logs to wandb |
| **Total** | **~66 min** | Dominated by encode_latents |

---

## GPU Concurrency Design

| Stage | GPU fraction | Concurrent actors (4 GPUs) | Why |
|-------|-------------|---------------------------|-----|
| embed | 0.25 | 16 | CLIP B/32 is small; saturate all 4 GPUs |
| score | 0.25 | 16 | CLIP L/14 + MLP fits at 0.25 GPU |
| caption | 1.0 | 4 | Qwen3-VL-8B needs full 40 GB A100 in bf16 |
| motion | CPU only | 16 (2 CPU each) | No GPU needed |
| caption_quality | 0.5 | 8 | CLIP B/32 text encode only |
| encode_latents | 0.5 | 8 | SD-VAE encoder |

**Decode path:** All Ray actors use PyAV CPU decode (`decode_video_actor()`). PyNvVideoCodec is unsafe for fractional-GPU actors — CUDA context racing causes `CUDA_ERROR_CONTEXT_IS_DESTROYED`. See [09-decode.md](09-decode.md).

---

## Output Directory Layout

```
$SCRATCH_DIR/                              # ~/scratch/video-curation (default)
├── raw_videos/
│   └── <label>/<yt_id>_<start>_<end>.mp4
├── embeddings/
│   └── <stem>.npz                         # mean_embedding [512], frame_embeddings [8,512]
├── md5_cache.json                         # {path: md5} — dedup rerun cache
├── faiss.index                            # persistent FAISS index
├── dedup_results.json                     # {kept, removed, exact_removed, threshold}
├── score_cache/<stem>.score.json
├── scores.json                            # [{path, mean_score, frame_scores, status}]
├── motion_cache/<stem>.motion.json
├── motion_scores.json                     # [{path, flow_magnitude, ssim_mean, motion_score}]
├── caption_cache/<stem>.caption.json
├── captions.json                          # [{path, caption, status}]
├── caption_quality_cache/<stem>.cq.json
├── caption_quality.json                   # [{path, clip_alignment, quality_score, ...}]
├── latents/<stem>.pt                      # float16 [4, T, H//8, W//8]
├── shards/shard_XXXXX.tar                 # {key}.mp4 + .txt + .json + .latent.pt
├── manifests/v1.json  v2.json
└── reports/embed_cluster_profile.html
```

---

## Utility Scripts

| Script | Purpose |
|--------|---------|
| `filter.py` | CLIP-based semantic filtering by text query (optional) |
| `stats.py` | Dataset composition analysis — charts, class balance, PCA |
| `decode.py` | Video decode library — GPU (standalone) and CPU (Ray actors) |
| `profile_run.py` | torchscope GPU profiling wrapper for the embed stage |
| `prefetch_models.py` | Pre-downloads all model weights on the login node |
| `setup_env.sh` | Conda environment creation and pip install |

---

## Cluster Quick-Start

```bash
# 1. Login node setup (once)
bash setup_env.sh
python prefetch_models.py   # ~16 GB: CLIP, LAION, Qwen3-VL-8B, SD-VAE
git clone https://github.com/YashJain14/torchscope.git ~/torchscope

# 2. Submit full pipeline
export WANDB_API_KEY=<your_key>
qsub run_curation.pbs

# 3. Resume from a specific stage
python dag.py --version v2 --limit 10000 --from_stage motion --num_gpus 4

# 4. GPU profiling (optional)
qsub run_profile.pbs
```
