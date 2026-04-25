# Video Curation Pipeline

End-to-end pipeline that downloads Kinetics-400 clips from S3, runs CLIP embeddings, deduplication, aesthetic scoring, VLM captioning, and writes training-ready WebDataset shards with versioned manifests.

**Hardware target:** NVIDIA A100-SXM4-40GB · CUDA 12.4 · PyTorch 2.6  
**Cluster:** NTU HPC · PBS scheduler · 4× A100 GPUs per node

---

## Pipeline Overview

```
Kinetics-400 S3 (CVDF)
       │
       ▼
[ingest.py]      Download tar.gz from S3, extract MP4s into <label>/ dirs
       │
       ▼
[embed.py]       CLIP ViT-B/32 embeddings        (Ray, 0.25 GPU/task → 16 concurrent)
       │
       ▼
[dedup.py]       MD5 exact dedup → FAISS near-dedup (cosine sim > 0.95)
       │
       ▼
[score.py]       LAION aesthetic scorer           (Ray, 0.25 GPU/task → 16 concurrent)
       │
       ▼
[caption.py]     Qwen3-VL-8B captioning          (Ray, 1.0 GPU/task → 4 concurrent)
       │           only videos passing score filter
       ▼
[shard.py]       WebDataset .tar shards (video + caption + score + embedding)
       │           only videos with score ≥ 4.5 AND non-empty caption
       ▼
[manifest.py]    Versioned dataset manifest (v1.json, v2.json, ...)
```

---

## Stage Details

### Stage 1 — `ingest.py` · Data Acquisition
Downloads Kinetics-400 `.tar.gz` files from the CVDF S3 mirror and extracts MP4 clips into `raw_videos/<label>/` folders.

- **Input:** Kinetics-400 split CSV + S3 URLs
- **Output:** `raw_videos/<label>/<yt_id>_<start>_<end>.mp4`
- **Why S3 over yt-dlp:** Pre-cut 10s clips, stable URLs, 3-5× faster download
- **Cache:** Skips videos already on disk

### Stage 2 — `embed.py` · CLIP Embeddings
Extracts semantic embeddings for every video using CLIP ViT-B/32. Samples 8 frames per video via PyAV (CPU decode), runs them through CLIP on GPU, stores mean + per-frame embeddings.

- **Input:** `raw_videos/`
- **Output:** `embeddings/<stem>.npz` — `mean_embedding [512]`, `frame_embeddings [8, 512]`
- **Model:** `openai/clip-vit-base-patch32`
- **Concurrency:** 16 actors (4 per GPU × 4 GPUs), `num_gpus=0.25` per actor
- **Cache:** Skips `.npz` files already written
- **Why ViT-B/32 not L/14:** 4× faster; B/32 is accurate enough for cosine similarity dedup

| Metric | Value |
|---|---|
| Videos processed | 10,000 |
| Cached (rerun) | 9,998 |
| Failed | 2 |
| Elapsed (cached rerun) | 22s |

### Stage 3 — `dedup.py` · Deduplication
Two-stage deduplication to remove exact and near-duplicate videos.

**Stage 1 — MD5 exact dedup:** Hashes every file byte-for-byte. O(N), catches re-uploads and identical files.

**Stage 2 — FAISS near-dedup:** Loads all CLIP embeddings, builds a cosine similarity index (`IndexFlatIP`), finds pairs with similarity > 0.95 using Union-Find clustering, keeps one representative per cluster.

- **Input:** `raw_videos/`, `embeddings/`
- **Output:** `dedup_results.json` `{kept, removed, exact_removed, threshold}`, `faiss.index` (persistent, reused on reruns)
- **Why two stages:** MD5 is O(N) and catches exact copies before the O(N²) FAISS search
- **Why IndexFlatIP:** L2-normalised vectors → inner product = cosine similarity; exact search, sufficient for <100k videos

| Metric | Value |
|---|---|
| Input videos | 9,998 |
| Exact duplicates removed | 0 |
| Near-duplicates removed | 54 (0.54%) |
| Final kept | 9,944 |
| Elapsed | 91s |

### Stage 4 — `score.py` · Aesthetic Scoring
Scores every video using the LAION aesthetic predictor — a small MLP trained on human aesthetic ratings from LAION-5B, operating on CLIP ViT-L/14 embeddings.

- **Input:** `raw_videos/`
- **Output:** `scores.json` `[{path, mean_score, frame_scores, status}]`, `score_cache/`
- **Model:** `openai/clip-vit-large-patch14` + `camenduru/improved-aesthetic-predictor`
- **Score range:** [0–10]; threshold 4.5 used for filtering
- **Concurrency:** 16 actors (4 per GPU × 4 GPUs), `num_gpus=0.25` per actor
- **Cache:** Per-video `.score.json` in `score_cache/`
- **Why ViT-L/14:** The aesthetic MLP was trained specifically on L/14 embeddings — must match backbone

| Metric | Value |
|---|---|
| Videos scored | 9,998 |
| Failed | 2 |
| Mean score | 4.26 |
| Passing (≥ 4.5) | 2,993 (29.9%) |
| Elapsed (cached rerun) | 23s |

> **Note:** Low pass rate is expected — Kinetics-400 is an action recognition dataset, not an aesthetic one. Clips are often shaky handheld footage collected from YouTube.

### Stage 5 — `caption.py` · VLM Captioning
Generates a natural language caption for each video that passed the aesthetic score filter using Qwen3-VL-8B-Instruct. Passes the raw video file directly to the model — no manual frame extraction needed.

- **Input:** `raw_videos/`, `scores.json` (for pre-filtering)
- **Output:** `captions.json` `[{path, caption, status}]`, `caption_cache/`
- **Model:** `Qwen/Qwen3-VL-8B-Instruct`
- **Settings:** `max_frames=8`, `max_pixels=128×32×32`, `max_new_tokens=128`
- **Concurrency:** 4 actors (1 per GPU), `num_gpus=1.0` per actor
- **Pre-filter:** Only captions videos with `mean_score ≥ 4.5` — saves ~70% of VLM compute
- **Cache:** Per-video `.caption.json` in `caption_cache/`
- **Why native video input:** Eliminates manual decode → frame sample → PIL pipeline; model handles frame sampling internally

| Metric | Value |
|---|---|
| Videos captioned | 2,993 |
| Successful | 2,993 (100%) |
| Failed | 0 |
| Elapsed | 187s (~3.1 min) |
| Throughput | ~16 videos/min across 4 GPUs |

### Stage 6 — `shard.py` · WebDataset Packaging
Assembles the final training dataset by packaging curated videos into WebDataset `.tar` shards. Only includes videos that pass all three filters: dedup kept + score ≥ 4.5 + non-empty caption.

Each sample in a shard contains:
- `<key>.mp4` — raw video bytes
- `<key>.txt` — Qwen3-VL caption
- `<key>.json` — metadata (path, label, aesthetic score, CLIP embedding)

- **Input:** `raw_videos/`, `captions.json`, `scores.json`, `dedup_results.json`, `embeddings/`
- **Output:** `shards/shard_XXXXX.tar`
- **Shard size:** 200 samples per shard
- **Why WebDataset:** Sequential reads saturate disk/network bandwidth; compatible with PyTorch DataLoader, HuggingFace datasets, NVIDIA DALI; streamable from S3/GCS

| Metric | Value |
|---|---|
| Videos written | 2,993 |
| Shards produced | 15 |
| Total size | 5,833 MB (5.8 GB) |
| Avg shard size | ~390 MB |
| Elapsed | 13s |

### Stage 7 — `manifest.py` · Versioned Manifest
Creates a versioned JSON record of the pipeline run for reproducibility. Records what went in, what came out, which thresholds were used, and which shard files exist.

- **Input:** `raw_videos/`, `dedup_results.json`, `scores.json`, `shards/`
- **Output:** `manifests/v1.json`
- **Supports:** `create`, `list`, `diff` subcommands for comparing dataset versions

| Metric | Value |
|---|---|
| Total videos | 10,000 |
| After dedup | 9,944 |
| Passing score | 2,993 |
| Shards | 15 files (5,833 MB) |

---

## v1 Pipeline Run Summary

Full pipeline on 10,000 Kinetics-400 val clips, 4× A100-SXM4-40GB:

| Stage | Runtime | Notes |
|---|---|---|
| ingest | — | S3 download (one-time) |
| embed | 22s | All cached from previous run |
| dedup | 91s | FAISS index built + saved |
| score | 23s | All cached from previous run |
| caption | 187s | 2,993 videos × 4 GPUs |
| shard | 13s | Pure I/O |
| manifest | ~5s | Pure compute |
| **Total** | **~6 min** | Mostly cached |

**Funnel:**
```
10,000 ingested
  └─ 9,998 embedded (2 decode failures)
      └─ 9,944 after dedup (54 near-duplicates removed)
          └─ 2,993 passing aesthetic score ≥ 4.5 (29.9%)
              └─ 2,993 captioned (100% success)
                  └─ 2,993 written to 15 WebDataset shards (5.8 GB)
```

---

## GPU Concurrency Design

| Stage | GPU fraction | Concurrent tasks (4 GPUs) | Why |
|---|---|---|---|
| embed | 0.25 | 16 | CLIP B/32 is small; pack 4 per GPU to saturate compute |
| score | 0.25 | 16 | Same — CLIP L/14 + MLP fits easily at 0.25 GPU |
| caption | 1.0 | 4 | Qwen3-VL-8B needs full 40 GB A100 in bf16 |

**Decode path:** All Ray actors use PyAV (CPU decode) via `decode_video_actor()`. GPU decode (PyNvVideoCodec) is unsafe for fractional-GPU actors due to CUDA context contention on shared NVDEC engines — CPU decode costs only a few ms vs hundreds of ms for model inference.

---

## Running on the Cluster

### Step 1 — Setup (login node, once)

```bash
bash setup_env.sh

# Pre-fetch model weights into the HF cache. Compute nodes have no
# internet (HF_HUB_OFFLINE=1), so this MUST run on the login node.
# Pulls CLIP ViT-B/32, ViT-L/14, LAION aesthetic predictor, Qwen3-VL-8B (~16 GB)
python prefetch_models.py

# Clone torchscope (no pip install needed — used via local path)
git clone https://github.com/YashJain14/torchscope.git ~/torchscope
```

### Step 2 — Run the full pipeline

```bash
qsub run_curation.pbs
```

Runs all 7 stages end-to-end. Logs to `curation_output.log`.  
Check status: `qstat -u $USER`  
Watch log: `tail -f curation_output.log`

### Step 3 — Resume from a specific stage

```bash
git pull  # always pull latest before running
python dag.py --version v1 --limit 10000 --from_stage caption --num_gpus 4
python dag.py --version v1 --limit 10000 --from_stage shard --num_gpus 4
python dag.py --version v1 --limit 10000 --from_stage manifest --num_gpus 4
```

### Step 4 — GPU profiling (optional)

```bash
qsub run_profile.pbs
```

Re-runs embed with [torchscope](https://github.com/YashJain14/torchscope) GPU profiling. Generates HTML reports in `$SCRATCH_DIR/reports/`.

```bash
scp user@cluster:~/scratch/video-curation/reports/*.html .
```

---

## Architecture Decisions

| Decision | Why |
|---|---|
| S3 CVDF ingest | Pre-cut clips, stable URLs, ~3-5× faster than yt-dlp; mirrors real production data lake patterns |
| Ray 0.25 GPU/task | Pack 4 tasks per GPU for embed/score; saturates all 4 A100s with 16 concurrent tasks |
| Two-stage dedup | MD5 catches exact copies cheaply O(N); FAISS near-dedup catches re-encoded duplicates |
| FAISS IndexFlatIP | L2-normalised vectors → inner product = cosine sim; exact search, fast for <100k videos |
| Persistent FAISS index | Saved to disk after build; loaded on rerun — avoids O(N²) rebuild for incremental datasets |
| Score before caption | Aesthetic scoring is cheap (~0.1s/video); VLM captioning is expensive (~3s/video). Filter first. |
| LAION aesthetic scorer | Pretrained MLP over CLIP ViT-L/14; predicts human aesthetic ratings [0–10] |
| Qwen3-VL-8B native video | Passes raw video file directly — no manual frame extraction; model handles sampling internally |
| Skip empty captions in shard | Ensures every training sample has both a quality score and a valid caption |
| WebDataset shards | Sequential reads saturate disk/network bandwidth; streamable from S3/GCS |
| Prefect orchestration | Zero-config local execution; DAG/task/retry model; per-stage caching via Prefect state |
| Versioned manifests | Any training data mix reproducible from a single JSON; supports diff between versions |
| torchscope profiling | GPU observability toolkit; proves 0.25 GPU/task allocation loads all 4 GPUs efficiently |

---

## Directory Layout

```
$SCRATCH_DIR/                         # ~/scratch/video-curation
├── raw_videos/
│   └── <label>/<yt_id>_<start>_<end>.mp4
├── embeddings/
│   └── <video_stem>.npz              # mean_embedding [512], frame_embeddings [8,512]
├── faiss.index                       # persistent FAISS index (reused on reruns)
├── dedup_results.json                # {kept, removed, exact_removed, threshold}
├── score_cache/
│   └── <video_stem>.score.json
├── scores.json                       # [{path, mean_score, frame_scores, status}]
├── caption_cache/
│   └── <video_stem>.caption.json
├── captions.json                     # [{path, caption, status}]
├── shards/
│   └── shard_00000.tar               # WebDataset: {key}.mp4 + {key}.txt + {key}.json
├── manifests/
│   └── v1.json
└── reports/
    ├── embed_cluster_profile.html    # torchscope cluster GPU report
    └── embed_driver_profile.html     # torchscope driver GPU timeline
```

---

## Individual Scripts

```bash
# Ingest from S3
python ingest.py --split val --out_dir $SCRATCH_DIR/raw_videos --limit 10

# Embed (CLIP ViT-B/32)
python embed.py --video_dir $SCRATCH_DIR/raw_videos --out_dir $SCRATCH_DIR/embeddings --num_gpus 4

# Dedup (MD5 + FAISS)
python dedup.py --video_dir $SCRATCH_DIR/raw_videos \
    --emb_dir $SCRATCH_DIR/embeddings \
    --threshold 0.95 \
    --index_path $SCRATCH_DIR/faiss.index

# Semantic filter (CLIP text query)
python filter.py --emb_dir $SCRATCH_DIR/embeddings --query "person playing sports" --top_k 200

# Score (LAION aesthetic)
python score.py --video_dir $SCRATCH_DIR/raw_videos --out $SCRATCH_DIR/scores.json --num_gpus 4

# Caption (Qwen3-VL-8B, score-filtered)
python caption.py --video_dir $SCRATCH_DIR/raw_videos \
    --out $SCRATCH_DIR/captions.json \
    --scores $SCRATCH_DIR/scores.json \
    --min_score 4.5 \
    --num_gpus 4

# Dataset stats + charts
python stats.py --video_dir $SCRATCH_DIR/raw_videos \
    --scores $SCRATCH_DIR/scores.json \
    --captions $SCRATCH_DIR/captions.json \
    --dedup $SCRATCH_DIR/dedup_results.json \
    --emb_dir $SCRATCH_DIR/embeddings

# Write WebDataset shards
python shard.py \
    --video_dir $SCRATCH_DIR/raw_videos \
    --captions  $SCRATCH_DIR/captions.json \
    --scores    $SCRATCH_DIR/scores.json \
    --dedup     $SCRATCH_DIR/dedup_results.json \
    --emb_dir   $SCRATCH_DIR/embeddings \
    --out_dir   $SCRATCH_DIR/shards \
    --min_score 4.5

# Versioned manifest
python manifest.py create --version v1 \
    --video_dir $SCRATCH_DIR/raw_videos \
    --dedup     $SCRATCH_DIR/dedup_results.json \
    --scores    $SCRATCH_DIR/scores.json \
    --shards    $SCRATCH_DIR/shards \
    --out       $SCRATCH_DIR/manifests/v1.json

python manifest.py diff --a $SCRATCH_DIR/manifests/v1.json --b $SCRATCH_DIR/manifests/v2.json
python manifest.py list --manifest_dir $SCRATCH_DIR/manifests
```

---

## Known Limitations

- Captioning is the bottleneck — Qwen3-VL-8B in bf16 needs the full 40 GB A100 per task
- FAISS `IndexFlatIP` is exact search, O(N²) at query time — for >500k videos swap to `IndexIVFFlat(nlist=1024)`
- Aesthetic score alone is a weak quality signal at scale — real pipelines add NSFW filtering, watermark detection, motion blur detection
- Kinetics-400 S3 hosting is maintained by CVDF — URLs are stable but the dataset requires accepting the Kinetics licence
