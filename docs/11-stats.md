# Utility — stats.py · Dataset Analysis

Dataset composition analysis. Loads scores, captions, embeddings, and dedup results to produce charts and statistics about the curated dataset. Not part of the main pipeline DAG — run after the pipeline completes to understand what was produced.

---

## What it does

Produces 6 types of analysis:

| Analysis                 | Description                                                   |
|--------------------------|---------------------------------------------------------------|
| Label distribution       | Clip count per Kinetics action category                       |
| Score histogram          | Aesthetic quality distribution with threshold lines           |
| Dedup summary            | How many near-duplicates were found and at what threshold     |
| Caption coverage         | How many clips have captions, empty-caption count             |
| CLIP embedding PCA       | 2D projection of semantic space coloured by label             |
| Underrepresented classes | Labels with fewer clips than `--min_clips` threshold          |

All stats are written to `dataset_stats.json`. Charts are saved as PNG files.

---

## Inputs

| Path                  | Description                             |
|-----------------------|-----------------------------------------|
| `raw_videos/**/*.mp4` | Source videos (for label counting)      |
| `scores.json`         | Scores from Stage 4                     |
| `captions.json`       | Captions from Stage 5                   |
| `dedup_results.json`  | Dedup results from Stage 3              |
| `embeddings/*.npz`    | CLIP embeddings from Stage 2 (for PCA)  |

All inputs are optional — stats are produced for whichever files exist.

---

## Outputs

| Path                            | Description                                        |
|---------------------------------|----------------------------------------------------|
| `dataset_stats.json`            | JSON with all computed statistics                  |
| `stats_score_histogram.png`     | Aesthetic score distribution chart                 |
| `stats_label_distribution.png`  | Top-30 action class clip counts bar chart          |
| `stats_embedding_pca.png`       | PCA of CLIP embeddings coloured by label           |

---

## CLI

```bash
python stats.py \
  --video_dir $SCRATCH_DIR/raw_videos \
  --scores    $SCRATCH_DIR/scores.json \
  --captions  $SCRATCH_DIR/captions.json \
  --dedup     $SCRATCH_DIR/dedup_results.json \
  --emb_dir   $SCRATCH_DIR/embeddings \
  --out_dir   $SCRATCH_DIR
```

| Argument      | Default           | Description                                      |
|---------------|-------------------|--------------------------------------------------|
| `--video_dir` | required          | Directory with raw MP4 files                     |
| `--scores`    | `data/scores.json`| Path to scores.json                              |
| `--captions`  | `data/captions.json` | Path to captions.json                         |
| `--dedup`     | `data/dedup_results.json` | Path to dedup_results.json              |
| `--emb_dir`   | `data/embeddings` | Directory with .npz files (for PCA)              |
| `--out_dir`   | `data`            | Output directory for JSON and PNG files          |
| `--min_clips` | `10`              | Flag classes with fewer clips than this          |

---

## Computed Statistics

### `dataset_stats.json` schema

```json
{
  "label_distribution": {"playing_guitar": 48, "air_drumming": 22, ...},
  "n_classes":          400,
  "total_clips":        10000,
  "underrepresented_classes": ["label_a", "label_b"],
  "scores": {
    "n":               9998,
    "mean":            4.261,
    "median":          4.18,
    "std":             0.82,
    "min":             1.34,
    "max":             8.91,
    "pct_above_4.5":   29.9,
    "pct_above_5.0":   12.4
  },
  "dedup": {
    "total":           9998,
    "kept":            9944,
    "removed":         54,
    "duplicate_pct":   0.5,
    "threshold":       0.95
  },
  "captions": {
    "total_videos":    10000,
    "captioned":       2993,
    "coverage_pct":    29.9,
    "empty":           0,
    "failed":          0
  }
}
```

### PCA Plot

Loads up to 2000 CLIP embeddings, fits a 2-component PCA, and plots each video as a coloured dot where the colour represents the Kinetics action label. Reveals how well CLIP separates different activity categories.

Requires `scikit-learn` — silently skipped if not installed.

---

## wandb Metrics Logged

| Metric                        | Description                            |
|-------------------------------|----------------------------------------|
| `stats/n_classes`             | Unique action categories               |
| `stats/total_clips`           | Total video clips                      |
| `stats/n_underrepresented`    | Classes below `min_clips` threshold    |
| `stats/score_mean`            | Mean aesthetic score                   |
| `stats/score_median`          | Median aesthetic score                 |
| `stats/score_std`             | Score standard deviation               |
| `stats/pct_above_4_5`         | % videos with score ≥ 4.5             |
| `stats/pct_above_5_0`         | % videos with score ≥ 5.0             |
| `stats/dedup_kept`            | Videos after dedup                     |
| `stats/dedup_removed`         | Near-duplicates removed                |
| `stats/duplicate_pct`         | Near-duplicate percentage              |
| `stats/captioned`             | Successfully captioned videos          |
| `stats/caption_coverage_pct`  | Caption coverage percentage            |
| `stats/stats_score_histogram` | wandb.Image of the score histogram PNG |
| `stats/stats_label_distribution` | wandb.Image of the label distribution PNG |
| `stats/stats_embedding_pca`   | wandb.Image of the PCA plot PNG        |
| `stats/label_distribution`    | wandb.Table for sortable inspection    |
