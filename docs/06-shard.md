# Stage 6 — shard.py · WebDataset Packaging

Assembles the final training dataset by packaging curated videos into WebDataset `.tar` shards. Only includes videos that pass all three filters: dedup kept + score ≥ 4.5 + non-empty caption.

---

## What it does

1. Loads `captions.json`, `scores.json`, and `dedup_results.json` into lookup dicts keyed by video path.
2. Applies the dedup filter: only processes videos in `dedup_results["kept"]`.
3. For each kept video (in sorted order):
   - Checks `mean_score ≥ min_score` — skips otherwise.
   - Checks caption is non-empty — skips otherwise.
   - Reads the raw video bytes from disk.
   - Loads the CLIP `mean_embedding` from the `.npz` file.
   - Opens a new shard tar every `shard_size` samples.
   - Writes `{key}.mp4`, `{key}.txt`, `{key}.json` into the current tar.
4. Closes the final tar and logs stats.

---

## Inputs

| Path                  | Description                                        |
|-----------------------|----------------------------------------------------|
| `raw_videos/**/*.mp4` | Source video files                                 |
| `captions.json`       | Captions from Stage 5                              |
| `scores.json`         | Aesthetic scores from Stage 4                      |
| `dedup_results.json`  | Kept video list from Stage 3                       |
| `embeddings/*.npz`    | CLIP embeddings from Stage 2 (for metadata)        |

---

## Outputs

| Path                     | Description                                    |
|--------------------------|------------------------------------------------|
| `shards/shard_XXXXX.tar` | WebDataset shard files                         |

### Shard contents per sample

Each sample uses a consistent key `{shard_idx:05d}_{sample_in_shard:04d}`:

| File          | Content                                                           |
|---------------|-------------------------------------------------------------------|
| `<key>.mp4`   | Raw video bytes                                                   |
| `<key>.txt`   | Qwen3-VL caption text (UTF-8)                                     |
| `<key>.json`  | Metadata: `{video_path, label, aesthetic_score, clip_embedding}`  |

---

## CLI

```bash
python shard.py \
  --video_dir  $SCRATCH_DIR/raw_videos \
  --captions   $SCRATCH_DIR/captions.json \
  --scores     $SCRATCH_DIR/scores.json \
  --dedup      $SCRATCH_DIR/dedup_results.json \
  --emb_dir    $SCRATCH_DIR/embeddings \
  --out_dir    $SCRATCH_DIR/shards \
  --shard_size 200 \
  --min_score  4.5
```

| Argument       | Default           | Description                                   |
|----------------|-------------------|-----------------------------------------------|
| `--video_dir`  | required          | Directory with raw MP4 files                  |
| `--captions`   | required          | Path to captions.json                         |
| `--scores`     | required          | Path to scores.json                           |
| `--dedup`      | required          | Path to dedup_results.json                    |
| `--emb_dir`    | `data/embeddings` | Directory with .npz files (for metadata)      |
| `--out_dir`    | `data/shards`     | Output directory for shard tars               |
| `--shard_size` | `200`             | Max samples per shard                         |
| `--min_score`  | `4.5`             | Aesthetic score threshold                     |

---

## Three-Gate Filter Logic

A video is written to a shard only if **all three** conditions are met:

```python
1. str(video_path) in kept_set          # passed dedup
2. mean_score >= min_score              # passed aesthetic filter
3. caption != ""                        # has a valid caption
```

Skipping empty captions ensures every training sample has both a quality signal (score) and a text label (caption). Samples missing either are not useful for text-conditioned generative training.

---

## Why WebDataset

Sequential `.tar` shards are the standard format for large-scale training dataloaders:

| Property          | Benefit                                                |
|-------------------|--------------------------------------------------------|
| Sequential reads  | Saturates disk and network bandwidth; no random seeks  |
| PyTorch compatible| Works with `webdataset.WebDataset`, `DataLoader`       |
| HuggingFace       | Compatible with `datasets.load_dataset("webdataset")`  |
| NVIDIA DALI       | Native WebDataset source support                       |
| S3/GCS streaming  | Shards can be streamed directly without full download  |
| Fixed shard size  | Easy to shuffle at the shard level, not sample level   |

---

## v1 Run Metrics

| Metric           | Value              |
|------------------|--------------------|
| Videos written   | 2,993              |
| Shards produced  | 15                 |
| Total size       | 5,833 MB (5.8 GB)  |
| Avg shard size   | ~390 MB            |
| Elapsed          | 13s                |

---

## wandb Metrics Logged

| Metric                       | Description                      |
|------------------------------|----------------------------------|
| `shard/n_shards`             | Number of shard files written    |
| `shard/n_videos_input`       | Videos after dedup filter        |
| `shard/total_bytes`          | Total output size in bytes       |
| `shard/total_mb`             | Total output size in MB          |
| `shard/elapsed_s`            | Wall time                        |

---

## Loading Shards for Training

```python
import webdataset as wds

dataset = (
    wds.WebDataset("shards/shard_{00000..00014}.tar")
    .decode("torchrgb")
    .to_tuple("mp4", "txt", "json")
)

for video_bytes, caption, metadata in dataset:
    # video_bytes: raw MP4 bytes
    # caption: str
    # metadata: dict with label, aesthetic_score, clip_embedding
    ...
```
