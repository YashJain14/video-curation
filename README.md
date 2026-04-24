# Video Curation Pipeline

End-to-end pipeline that downloads Kinetics-400 clips from S3, runs CLIP embeddings, deduplication, aesthetic scoring, VLM captioning, and writes training-ready WebDataset shards with versioned manifests.

**Hardware target:** NVIDIA A100-SXM4-40GB · CUDA 12.4 · PyTorch 2.6  
**Cluster:** NTU HPC · PBS scheduler · 4× A100 GPUs per node

## Pipeline

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
[caption.py]     Qwen2.5-VL-7B captioning         (Ray, 1.0 GPU/task → 4 concurrent)
       │
       ▼
[shard.py]       WebDataset .tar shards (video + caption + score + embedding)
       │
       ▼
[manifest.py]    Versioned dataset manifest (v1.json, v2.json, ...)
```

## Running on the Cluster

### Step 1 — Setup (login node, once)

```bash
bash setup_env.sh

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

### Step 3 — Run GPU profiling (after pipeline completes)

```bash
qsub run_profile.pbs
```

Re-runs the embed stage wrapped with [torchscope](https://github.com/YashJain14/torchscope) GPU profiling. No pip install needed — `--torchscope ~/torchscope` adds it to `sys.path` at runtime.

Generates two HTML reports in `$SCRATCH_DIR/reports/`:
- `embed_cluster_profile.html` — per-worker GPU utilisation + cluster imbalance stats
- `embed_driver_profile.html`  — driver-side GPU timeline with throughput metadata

Copy reports to your laptop:
```bash
scp user@cluster:~/scratch/video-curation/reports/*.html .
```

### Skip completed stages

```bash
# Resume from a specific stage (skips everything before it)
python dag.py --version v1 --limit 500 --from_stage score
python dag.py --version v1 --limit 500 --from_stage caption
python dag.py --version v1 --limit 500 --from_stage shard
```

## PBS Scripts

| Script | What it runs | Walltime |
|---|---|---|
| `run_curation.pbs` | Full 7-stage pipeline via `dag.py` | 4h |
| `run_profile.pbs` | Embed stage with torchscope GPU profiling | 1h |

## Individual Scripts

```bash
# Ingest from S3
python ingest.py --split val --out_dir $SCRATCH_DIR/raw_videos --limit 10

# Embed (CLIP ViT-B/32)
python embed.py --video_dir $SCRATCH_DIR/raw_videos --out_dir $SCRATCH_DIR/embeddings --num_gpus 4

# Dedup (MD5 + FAISS)
python dedup.py --video_dir $SCRATCH_DIR/raw_videos --emb_dir $SCRATCH_DIR/embeddings --threshold 0.95

# Semantic filter (CLIP text query)
python filter.py --emb_dir $SCRATCH_DIR/embeddings --query "person playing sports" --top_k 200

# Score (LAION aesthetic)
python score.py --video_dir $SCRATCH_DIR/raw_videos --out $SCRATCH_DIR/scores.json --num_gpus 4

# Caption (Qwen2.5-VL-7B)
python caption.py --video_dir $SCRATCH_DIR/raw_videos --out $SCRATCH_DIR/captions.json --num_gpus 4

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

## Architecture Decisions

| Decision | Why |
|---|---|
| S3 CVDF ingest | Pre-cut clips, stable URLs, ~3-5× faster than yt-dlp; mirrors real production data lake patterns |
| Ray 0.25 GPU/task | Pack 4 tasks per GPU for embed/score; saturates all 4 A100s with 16 concurrent tasks |
| Two-stage dedup | MD5 catches exact copies cheaply (O(N)); FAISS near-dedup catches re-encoded duplicates |
| FAISS IndexFlatIP | L2-normalised vectors → inner product = cosine sim; exact search, fast for <100k videos |
| LAION aesthetic scorer | Pretrained MLP over CLIP ViT-L/14; predicts human aesthetic ratings [0–10] |
| Qwen2.5-VL-7B | Strong multi-frame video understanding; 1 full GPU/task for bf16 inference |
| WebDataset shards | Sequential reads saturate disk/network bandwidth; streamable from S3/GCS |
| Prefect orchestration | Zero-config local execution; same DAG/task/retry model as Airflow |
| Versioned manifests | Any training data mix reproducible from a single JSON; supports diff between versions |
| torchscope profiling | GPU observability toolkit; proves 0.25 GPU/task allocation loads all 4 GPUs efficiently |

## Scratch Directory Layout

```
$SCRATCH_DIR/                         # ~/scratch/video-curation
├── raw_videos/
│   └── <label>/<yt_id>_<start>_<end>.mp4
├── embeddings/
│   └── <video_stem>.npz              # mean_embedding [512], frame_embeddings [N,512]
├── dedup_results.json                # {kept, removed, exact_removed, threshold}
├── scores.json                       # [{path, mean_score, frame_scores, status}]
├── captions.json                     # [{path, caption, status}]
├── shards/
│   └── shard_00000.tar               # WebDataset: {key}.mp4 + {key}.txt + {key}.json
├── manifests/
│   └── v1.json
└── reports/
    ├── embed_cluster_profile.html    # torchscope cluster GPU report
    └── embed_driver_profile.html     # torchscope driver GPU timeline
```

## GPU Concurrency

| Stage | GPU fraction | Concurrent tasks (4 GPUs) |
|---|---|---|
| embed | 0.25 | 16 |
| score | 0.25 | 16 |
| caption | 1.0 | 4 |

## Known Limitations

- Captioning is the bottleneck — Qwen2.5-VL-7B in bf16 needs the full 40 GB A100 per task.
- FAISS `IndexFlatIP` is exact search, O(N²) at query time. For >500k videos swap to `IndexIVFFlat(nlist=1024)`.
- Kinetics-400 S3 hosting is maintained by CVDF — URLs are stable but the dataset requires accepting the Kinetics licence.
