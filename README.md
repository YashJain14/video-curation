# Video Curation Pipeline

End-to-end pipeline that takes raw video clips and produces training-ready WebDataset shards with captions, aesthetic scores, and deduplication.

**Hardware target:** NVIDIA A100-SXM4-40GB · CUDA 12.4 · PyTorch 2.6

## Pipeline

```
Raw Videos (Kinetics-400)
       │
       ▼
[ingest.py]      yt-dlp parallel download
       │
       ▼
[embed.py]       CLIP ViT-B/32 embeddings  (Ray parallel, 0.25 GPU/task)
       │
       ▼
[dedup.py]       FAISS IndexFlatIP near-dedup (cosine sim > 0.95)
       │
       ▼
[score.py]       LAION aesthetic scorer      (Ray parallel, 0.25 GPU/task)
       │
       ▼
[caption.py]     Qwen2.5-VL-7B captioning    (Ray parallel, 1 GPU/task)
       │
       ▼
[shard.py]       WebDataset .tar shards (video + caption + metadata)
       │
       ▼
[manifest.py]    Versioned dataset manifest (v1.json, v2.json, ...)
```

## Quick Start

```bash
# 1. Setup environment (login node)
bash setup_env.sh

# 2. Download Kinetics-400 val CSV
bash download_kinetics_csv.sh

# 3. Submit full pipeline
qsub run_curation.pbs

# 4. Or run individual stages
python dag.py --version v1 --limit 500 --from_stage ingest
python dag.py --version v1 --limit 500 --from_stage embed    # skip ingest
python dag.py --version v1 --limit 500 --from_stage score    # skip ingest+embed+dedup
```

## Individual Scripts

```bash
# Download clips
python ingest.py --csv data/kinetics400_val.csv --out_dir data/raw_videos --limit 500

# Embed (CLIP)
python embed.py --video_dir data/raw_videos --out_dir data/embeddings --frames_per_video 8

# Dedup (FAISS)
python dedup.py --emb_dir data/embeddings --threshold 0.95

# Score (aesthetic)
python score.py --video_dir data/raw_videos --out data/scores.json

# Caption (Qwen2.5-VL)
python caption.py --video_dir data/raw_videos --out data/captions.json --frames_per_video 4

# Write shards
python shard.py \
    --video_dir data/raw_videos \
    --captions  data/captions.json \
    --scores    data/scores.json \
    --dedup     data/dedup_results.json \
    --out_dir   data/shards \
    --min_score 4.5

# Create manifest
python manifest.py create \
    --version v1 \
    --video_dir data/raw_videos \
    --dedup data/dedup_results.json \
    --scores data/scores.json \
    --shards data/shards \
    --out data/manifests/v1.json

# Compare versions
python manifest.py diff --a data/manifests/v1.json --b data/manifests/v2.json

# List all versions
python manifest.py list --manifest_dir data/manifests
```

## Architecture Decisions

| Decision | Why |
|---|---|
| Ray for embed/score/caption | Parallelise GPU work across videos; each task gets a fraction of the GPU |
| FAISS IndexFlatIP | L2-normalised embeddings → inner product == cosine sim; exact search, fast enough for <100k videos |
| LAION aesthetic scorer | Pretrained MLP over CLIP ViT-L/14; predicts human aesthetic ratings; threshold >4.5 removes low-quality content |
| Qwen2.5-VL for captions | Strong multi-frame video understanding; open weights; outputs natural language captions for text-conditioned training |
| WebDataset shards | Sequential reads saturate disk/network bandwidth; compatible with PyTorch, DALI, HuggingFace; streamable from S3/GCS |
| Prefect for orchestration | Zero-config local execution; same DAG concept as Airflow; easy to migrate |
| Versioned manifests | Any training data mix is reproducible from a single JSON file |

## Data Directory Layout

```
data/
├── kinetics400_val.csv
├── raw_videos/
│   └── <label>/<yt_id>_<start>.mp4
├── embeddings/
│   └── <video_stem>.npz          # mean_embedding [512], frame_embeddings [N,512]
├── dedup_results.json             # {kept: [...], removed: [...]}
├── scores.json                    # [{path, mean_score, frame_scores}, ...]
├── captions.json                  # [{path, caption}, ...]
├── shards/
│   └── shard_00000.tar           # WebDataset: {key}.mp4 + {key}.txt + {key}.json
└── manifests/
    ├── v1.json
    └── v2.json
```

## Known Limitations

- `caption.py` uses `num_gpus=1` per Ray task — Qwen2.5-VL-7B needs the full GPU. Captioning is the bottleneck; run it last.
- FAISS `IndexFlatIP` is exact and O(N²) at query time. For >500k videos, swap to `IndexIVFFlat` with `nlist=1024`.
- Kinetics-400 YouTube links go dead over time (~20-30% unavailable). `ingest.py` logs failures and continues.
