# Video Curation Pipeline

End-to-end pipeline that downloads Kinetics-400 clips from S3, runs CLIP embeddings, deduplication, aesthetic scoring, motion quality scoring, VLM captioning, caption quality analysis, VAE latent pre-encoding, and writes training-ready WebDataset shards with versioned manifests.

**Hardware target:** NVIDIA A100-SXM4-40GB · CUDA 12.4 · PyTorch 2.6  
**Cluster:** NSCC HPC · PBS scheduler · 4× A100 GPUs per node

---

## Pipeline DAG

```
Kinetics-400 S3 (CVDF)
       │
       ▼
[ingest.py]           Download tar.gz from S3, extract MP4s
       │
       ▼
[embed.py]            CLIP ViT-B/32 embeddings         (Ray, 0.25 GPU/task → 16 concurrent)
       │
       ▼
[dedup.py]            MD5 exact + FAISS near-dedup      (MD5 cache + persistent FAISS index)
       │
       ▼
[score.py]            LAION aesthetic scorer            (Ray, 0.25 GPU/task → 16 concurrent)
       │
       ▼
[motion.py]           Optical flow + SSIM quality       (CPU Ray, 16 workers, cached)
       │               filters before VLM — avoids wasting Qwen3-VL on bad clips
       ▼
[caption.py]          Qwen3-VL-8B captioning            (Ray, 1.0 GPU/task → 4 concurrent)
       │               only videos passing aesthetic + motion gates
       ▼
[caption_quality.py]  CLIP text-image alignment         (Ray, 0.5 GPU/task → 8 concurrent)
       │               verifies VLM output quality
       ▼
[encode_latents.py]   SD-VAE latent pre-encoding        (Ray, 0.5 GPU/task → 8 concurrent)
       │               ~15–30% training GPU savings
       ▼
[shard.py]            WebDataset .tar shards            (video + caption + scores + latent)
       │               gates on all three quality signals
       ▼
[manifest.py]         Versioned dataset manifest        (full funnel stats + wandb)
```

Orchestrated by `dag.py` (Prefect flow). Resume from any stage with `--from_stage`. Run v1-only (skip motion/caption_quality/encode_latents) with `--no_v2`.

---

## Run Results — 10,000 Kinetics-400 val clips, 4× A100-SXM4-40GB

**Quality funnel:**
```
10,000 ingested
  └─ 9,998 embedded  (2 decode failures — corrupt videos)
      └─ 9,944 after dedup  (54 near-duplicates removed, cosine sim > 0.95)
          └─ 2,993 passing aesthetic ≥ 4.5  (29.9% — expected for action dataset)
              └─ 2,396 passing motion ≥ 5.0  (80.1% of aesthetic-passing)
                  └─ 2,396 captioned  (597 fewer VLM calls vs filtering on aesthetic alone)
                      └─ 2,161 passing caption quality ≥ 0.65  (90.2% of captioned)
                          └─ 2,161 latents encoded
                              └─ 2,155 written to 11 WebDataset shards  (4.1 GB)
```

## GPU Concurrency Design

| Stage | GPU fraction | Concurrent actors (4 GPUs) | Why |
|-------|-------------|---------------------------|-----|
| embed | 0.25 | 16 | CLIP B/32 is small; saturate all 4 GPUs |
| score | 0.25 | 16 | CLIP L/14 + MLP fits at 0.25 GPU |
| caption | 1.0 | 4 | Qwen3-VL-8B needs full 40 GB A100 in bf16 |
| motion | CPU only | 16 (2 CPU each) | No GPU needed; saturates CPU cores |
| caption_quality | 0.5 | 8 | CLIP B/32 text encode only |
| encode_latents | 0.5 | 8 | SD-VAE encoder |

**Decode path:** All Ray actors use PyAV CPU decode (`decode_video_actor()`). GPU decode (PyNvVideoCodec) causes `CUDA_ERROR_CONTEXT_IS_DESTROYED` crashes when multiple fractional-GPU actors share one physical GPU — see [docs/09-decode.md](docs/09-decode.md) for the full investigation.

---

## Setup & Running

### One-time setup (login node)

```bash
bash setup_env.sh
python prefetch_models.py   # ~16 GB: CLIP, LAION predictor, Qwen3-VL-8B, SD-VAE
```

Compute nodes have no internet (`HF_HUB_OFFLINE=1`). All model weights must be prefetched.

### Full pipeline

```bash
export WANDB_API_KEY=<your_key>
qsub run_curation.pbs
```

### Resume from a stage

```bash
python dag.py --version v2 --limit 10000 --from_stage motion --num_gpus 4
python dag.py --version v2 --limit 10000 --from_stage caption_quality --num_gpus 4
python dag.py --version v2 --limit 10000 --from_stage shard --num_gpus 4

# Skip v2 stages (v1 pipeline only)
python dag.py --version v1 --limit 10000 --from_stage caption --num_gpus 4 --no_v2
```

Valid `--from_stage`: `ingest embed dedup score motion caption caption_quality encode_latents shard manifest`

---

## Output Layout

```
$SCRATCH_DIR/                              # ~/scratch/video-curation
├── raw_videos/<label>/<yt_id>_<s>_<e>.mp4
├── embeddings/<stem>.npz                  # mean_embedding [512], frame_embeddings [8,512]
├── md5_cache.json                         # {path: md5} — speeds up dedup reruns
├── faiss.index                            # persistent FAISS index
├── dedup_results.json
├── score_cache/<stem>.score.json
├── scores.json
├── motion_cache/<stem>.motion.json
├── motion_scores.json                     # [{path, flow_magnitude, ssim_mean, motion_score}]
├── caption_cache/<stem>.caption.json
├── captions.json
├── caption_quality_cache/<stem>.cq.json
├── caption_quality.json                   # [{path, clip_alignment, quality_score, ...}]
├── latents/<stem>.pt                      # float16 [4, T, H//8, W//8]
├── shards/shard_00000.tar                 # {key}.mp4 + .txt + .json + .latent.pt
├── manifests/v1.json  v2.json
└── reports/embed_cluster_profile.html
```

---

## v1 → v2 Upgrade

v1 filtered on aesthetic score only and had no latent pre-encoding. v2 adds three stages that close the gap between "cleaned dataset" and "what video diffusion training actually needs":

| Gap in v1 | v2 fix |
|-----------|--------|
| Aesthetic score doesn't see temporal problems (frozen clips, scene cuts, shaky footage) | `motion.py` — Farneback optical flow + SSIM composite score |
| VLM captions unverified — hallucinations and vague descriptions go straight to shards | `caption_quality.py` — CLIP text-image alignment gate |
| VAE encoding repeated every training step across all epochs | `encode_latents.py` — pre-encode once, read float16 tensors at training time |
| All ~3k aesthetic-passing videos sent to Qwen3-VL regardless of temporal quality | Motion gate runs before caption — 597 fewer VLM calls |

**v1 → v2 diff (actual output from `manifest.py diff`):**
```
  Total videos                    10000 →        10000
  Dedup kept                       9944 →         9944
  Scoring passing                  2993 →         2993
  Motion passing                      — →         7665  (≥ 5.0)
  Caption qual passing                — →         2161  (≥ 0.65)
  Shards                             15 →           11  (Δ -4)
  Total samples                    2993 →         2155
  Total MB                       5833.7 →       4104.9  (Δ -1728.8 MB)
```

See [docs/00-overview.md](docs/00-overview.md) for full architecture documentation.

---

## Models

| Model | Repo | Used in |
|-------|------|---------|
| CLIP ViT-B/32 | `openai/clip-vit-base-patch32` | `embed.py`, `caption_quality.py` |
| CLIP ViT-L/14 | `openai/clip-vit-large-patch14` | `score.py` |
| LAION aesthetic MLP | `camenduru/improved-aesthetic-predictor` | `score.py` |
| Qwen3-VL-8B | `Qwen/Qwen3-VL-8B-Instruct` | `caption.py` |
| SD-VAE | `stabilityai/sd-vae-ft-mse` | `encode_latents.py` |

---

## Key Constraints

- **numpy < 2** — `faiss-gpu` is built against NumPy 1.x ABI
- **opencv-python-headless < 4.11** — versions ≥ 4.11 use NumPy 2 ABI
- **WANDB_API_KEY must be exported before `qsub`** — compute nodes have no interactive login
- **Aesthetic predictor weights** — must use `camenduru/improved-aesthetic-predictor`; other repos 404
