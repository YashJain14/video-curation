# Stage 1 — ingest.py · Data Acquisition

Downloads Kinetics-400 `.tar.gz` files from the CVDF S3 mirror and extracts MP4 clips into `raw_videos/<label>/` folders.

---

## What it does

1. Fetches the S3 path list (e.g. `k400_val_path.txt`) — a plain-text file listing all tar.gz URLs for the chosen split.
2. Downloads the annotation CSV from S3 to build a `{youtube_id → label}` lookup.
3. Downloads `--limit` tar files in parallel using `ThreadPoolExecutor`.
4. Extracts each tar in-place: clips go to `out_dir/<label>/<yt_id>_<start>_<end>.mp4`.
5. Writes a `.done` marker for each tar so reruns skip already-extracted archives.
6. Fixes any clips that landed in `unknown/` (missing label at extraction time) by re-labelling them from the CSV.

---

## Inputs

| Source              | Description                                      |
|---------------------|--------------------------------------------------|
| CVDF S3             | `s3://kinetics/400/{split}/k400_{split}_path.txt`|
| Annotation CSV      | `s3://kinetics/400/annotations/{split}.csv`      |

S3 access requires no AWS credentials — all URLs are public HTTP.

---

## Outputs

| Path                                          | Description                |
|-----------------------------------------------|----------------------------|
| `raw_videos/<label>/<yt_id>_<start>_<end>.mp4`| Extracted 10s MP4 clips    |
| `raw_videos/.<part_name>.done`                | Extraction done-markers    |
| `$SCRATCH_DIR/kinetics400_{split}.csv`        | Cached annotation CSV      |

---

## CLI

```bash
python ingest.py --split val --out_dir $SCRATCH_DIR/raw_videos --limit 10
python ingest.py --split train --out_dir $SCRATCH_DIR/raw_videos --limit 50
```

| Argument   | Default | Description                                     |
|------------|---------|--------------------------------------------------|
| `--split`  | `val`   | Dataset split: `train`, `val`, or `test`         |
| `--out_dir`| `$SCRATCH_DIR/raw_videos` | Output directory              |
| `--limit`  | `10`    | Max tar files to download (~50 clips each)       |
| `--workers`| `8`     | Parallel download threads                        |

---

## Key Implementation Details

### Tar extraction and label assignment

Kinetics tar files have a flat layout: `./` contains `<yt_id>_<start>_<end>.mp4` filenames. The label is not encoded in the filename — it must be looked up from the annotation CSV via the YouTube ID prefix.

```python
parts  = clip_name.replace(".mp4", "").rsplit("_", 2)
yt_id  = parts[0] if len(parts) == 3 else clip_name
label  = label_lookup.get(yt_id, "unknown")
```

Clips with no matching CSV entry go to `unknown/` and are re-labelled in a second pass (`relabel_existing`).

### Caching / idempotency

Each tar file has a `.done` marker written after successful extraction. On reruns, files with an existing marker are skipped without re-downloading.

### Parallelism

`ThreadPoolExecutor` with `--workers` threads (default 8). I/O-bound work — threads are appropriate here; the GIL doesn't matter for network and disk operations.

---

## Why S3 over yt-dlp

| Dimension       | S3 CVDF                         | yt-dlp                          |
|----------------|----------------------------------|---------------------------------|
| Speed           | ~3–5× faster (no throttling)    | YouTube rate-limits downloads   |
| Reliability     | Stable permanent URLs            | ~5–10% dead links               |
| Pre-processing  | Pre-cut 10s clips (no ffmpeg)    | Requires trim after download    |
| Authentication  | No credentials needed            | Requires handling age gates     |
| Production fit  | Mirrors real data lake on S3/GCS | Doesn't                         |

---

## wandb Metrics Logged

| Metric                  | Description                        |
|-------------------------|------------------------------------|
| `ingest/tars_ok`        | Successfully downloaded tar files  |
| `ingest/tars_cached`    | Skipped (already extracted)        |
| `ingest/tars_failed`    | Failed downloads                   |
| `ingest/total_clips`    | Total MP4 clips extracted          |
| `ingest/n_labels`       | Unique Kinetics action classes     |
| `ingest/elapsed_s`      | Wall time                          |
| `ingest/clips_per_s`    | Download throughput                |
