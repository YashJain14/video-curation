# Utility — filter.py · Semantic Filtering

CLIP-based semantic filtering of video clips. Given a text query, embed it with CLIP and keep only videos whose mean embedding has cosine similarity ≥ threshold against the query. Optional stage — not part of the main pipeline DAG, but usable standalone to select domain-specific subsets.

---

## What it does

1. Loads precomputed CLIP embeddings from `embeddings/*.npz` (produced by Stage 2).
2. Encodes the text query with CLIP ViT-B/32's text encoder.
3. Computes cosine similarity between the text embedding and every video embedding (both are L2-normalised, so this is a dot product).
4. Keeps videos with similarity ≥ threshold (or inverts the filter with `--invert`).
5. Sorts kept videos by descending similarity.
6. Writes results to `filter_results.json`.

---

## Inputs

| Path                | Description                               |
|---------------------|-------------------------------------------|
| `embeddings/*.npz`  | CLIP embeddings from Stage 2              |

No GPU needed for loading embeddings — all numpy operations.

---

## Outputs

| Path                    | Description                                        |
|-------------------------|----------------------------------------------------|
| `filter_results.json`   | `{query, threshold, kept: [{path, similarity}], removed: [path]}` |

---

## CLI

```bash
# Keep videos similar to "outdoor sports"
python filter.py \
  --emb_dir   $SCRATCH_DIR/embeddings \
  --query     "outdoor sports action" \
  --threshold 0.2

# Remove static/empty scenes (inverted filter)
python filter.py \
  --query    "static empty room" \
  --invert

# Domain-specific subset: cooking videos
python filter.py \
  --query    "person cooking food in kitchen" \
  --threshold 0.22
```

| Argument      | Default                     | Description                                  |
|---------------|-----------------------------|----------------------------------------------|
| `--query`     | required                    | Natural language description of target videos|
| `--emb_dir`   | `data/embeddings`           | Directory with .npz files                    |
| `--threshold` | `0.2`                       | Cosine similarity cutoff                     |
| `--invert`    | `False`                     | Keep videos BELOW threshold instead          |
| `--out`       | `data/filter_results.json`  | Output JSON path                             |
| `--device`    | `cuda:0` if available       | Device for CLIP text encoding                |

---

## CLIP Similarity Range

CLIP ViT-B/32 cosine similarity between a text query and a video frame embedding typically falls in the range **0.15–0.35**. This is narrower than image-to-image similarity because text and visual embeddings live in different parts of the shared space.

Practical thresholds:
- `0.15` — very broad filter (almost everything)
- `0.20` — moderate filter (recommended starting point)
- `0.25` — stricter filter
- `0.30+` — very specific matches only

---

## Use Cases

```bash
# Select action-heavy content
python filter.py --query "person actively performing an athletic move" --threshold 0.22

# Select cooking domain
python filter.py --query "person cooking food in kitchen" --threshold 0.22

# Remove low-motion scenes (invert: keep what's NOT static)
python filter.py --query "static empty room with no people" --invert --threshold 0.18
```

---

## wandb Metrics Logged

| Metric                | Description                              |
|-----------------------|------------------------------------------|
| `filter/total`        | Total videos evaluated                   |
| `filter/kept`         | Videos passing the filter                |
| `filter/removed`      | Videos removed by the filter             |
| `filter/kept_pct`     | Percentage kept                          |
| `filter/sim_mean`     | Mean cosine similarity across all videos |
| `filter/sim_median`   | Median cosine similarity                 |
| `filter/sim_max`      | Maximum cosine similarity                |
| `filter/sim_min`      | Minimum cosine similarity                |
| `filter/sim_hist`     | Full similarity distribution (histogram) |
| `filter/elapsed_s`    | Wall time                                |
