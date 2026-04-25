# Pipeline Overview

End-to-end video curation pipeline that downloads Kinetics-400 clips from S3, runs CLIP embeddings, deduplication, aesthetic scoring, VLM captioning, and writes training-ready WebDataset shards with versioned manifests.

**Hardware target:** NVIDIA A100-SXM4-40GB · CUDA 12.4 · PyTorch 2.6  
**Cluster:** NTU HPC · PBS scheduler · 4× A100 GPUs per node

---

## Pipeline DAG

```
Kinetics-400 S3 (CVDF)
       │
       ▼
[Stage 1: ingest.py]     Download tar.gz from S3, extract MP4s into <label>/ dirs
       │
       ▼
[Stage 2: embed.py]      CLIP ViT-B/32 embeddings        (Ray, 0.25 GPU/task → 16 concurrent)
       │
       ▼
[Stage 3: dedup.py]      MD5 exact dedup → FAISS near-dedup (cosine sim > 0.95)
       │
       ▼
[Stage 4: score.py]      LAION aesthetic scorer           (Ray, 0.25 GPU/task → 16 concurrent)
       │
       ▼
[Stage 5: caption.py]    Qwen3-VL-8B captioning          (Ray, 1.0 GPU/task → 4 concurrent)
       │                   only videos passing score filter
       ▼
[Stage 6: shard.py]      WebDataset .tar shards (video + caption + score + embedding)
       │                   only videos with score ≥ 4.5 AND non-empty caption
       ▼
[Stage 7: manifest.py]   Versioned dataset manifest (v1.json, v2.json, ...)
```

Orchestration is handled by `dag.py` (Prefect flow). Each stage can be run standalone or the pipeline can be resumed from any stage using `--from_stage`.

---

## v1 Run Results — 10,000 Kinetics-400 val clips, 4× A100

**Funnel:**
```
10,000 ingested
  └─ 9,998 embedded (2 decode failures)
      └─ 9,944 after dedup (54 near-duplicates removed)
          └─ 2,993 passing aesthetic score ≥ 4.5 (29.9%)
              └─ 2,993 captioned (100% success)
                  └─ 2,993 written to 15 WebDataset shards (5.8 GB)
```

**Runtime:**

| Stage   | Runtime | Notes                             |
|---------|---------|-----------------------------------|
| ingest  | —       | S3 download (one-time)            |
| embed   | 22s     | All cached from previous run      |
| dedup   | 91s     | FAISS index built + saved         |
| score   | 23s     | All cached from previous run      |
| caption | 187s    | 2,993 videos × 4 GPUs             |
| shard   | 13s     | Pure I/O                          |
| manifest| ~5s     | Pure compute                      |
| **Total**| **~6 min** | Mostly cached                 |

---

## GPU Concurrency Design

| Stage   | GPU fraction | Concurrent tasks (4 GPUs) | Why                                          |
|---------|-------------|---------------------------|----------------------------------------------|
| embed   | 0.25        | 16                        | CLIP B/32 is small; pack 4 per GPU            |
| score   | 0.25        | 16                        | CLIP L/14 + MLP fits at 0.25 GPU             |
| caption | 1.0         | 4                         | Qwen3-VL-8B needs full 40 GB A100 in bf16    |

**Decode path:** All Ray actors use PyAV (CPU decode) via `decode_video_actor()`. GPU decode (PyNvVideoCodec) is unsafe for fractional-GPU actors due to CUDA context contention on shared NVDEC engines.

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

## Utility Scripts

| Script              | Purpose                                                       |
|---------------------|---------------------------------------------------------------|
| `filter.py`         | CLIP-based semantic filtering by text query (optional stage)  |
| `stats.py`          | Dataset composition analysis — charts, class balance, PCA     |
| `decode.py`         | Video decode library — GPU (standalone) and CPU (Ray actors)  |
| `profile_run.py`    | torchscope GPU profiling wrapper for the embed stage          |
| `prefetch_models.py`| Pre-downloads all model weights on the login node             |
| `setup_env.sh`      | Conda environment creation and pip install                    |

---

## Cluster Quick-Start

```bash
# 1. Login node setup (once)
bash setup_env.sh
python prefetch_models.py
git clone https://github.com/YashJain14/torchscope.git ~/torchscope

# 2. Submit full pipeline
qsub run_curation.pbs

# 3. Resume from a specific stage
python dag.py --version v1 --limit 10000 --from_stage caption --num_gpus 4

# 4. GPU profiling (optional)
qsub run_profile.pbs
```
