# Stage 7 — manifest.py · Versioned Manifest

Creates a versioned JSON record of the pipeline run for reproducibility. Records what went in, what came out, which thresholds were used, and which shard files exist. Supports `create`, `list`, and `diff` subcommands.

---

## What it does

### `create` subcommand

1. Walks `raw_videos/` to count source videos.
2. Reads `dedup_results.json` for dedup stats (kept, removed, threshold).
3. Reads `scores.json` for score stats (mean, count passing threshold).
4. Walks `shards/` to enumerate shard files and sum their byte sizes.
5. Writes a single `manifests/<version>.json` with all of the above.

### `list` subcommand

Prints a tabular summary of all `.json` files in `manifests/`:
```
Version      Created                 Videos    Kept   Shards     Size MB
v1           2025-04-25T10:30:00     10000    9944       15      5833.0
v2           2025-04-26T14:15:00     20000   19800       30     11200.0
```

### `diff` subcommand

Compares two manifest versions side-by-side and shows deltas:
```
Diff: v1 → v2
  Total videos                   10000 →        20000  (Δ +10000.0)
  Dedup kept                      9944 →        19800  (Δ +9856.0)
  Scoring passing                 2993 →         5900  (Δ +2907.0)
  Shards                            15 →           30  (Δ +15.0)
  Total MB                      5833.0 →       11200.0  (Δ +5367.0)
```

---

## Inputs

| Path                  | Description                              |
|-----------------------|------------------------------------------|
| `raw_videos/**/*.mp4` | Source video files (counted)             |
| `dedup_results.json`  | Dedup stats from Stage 3                 |
| `scores.json`         | Score stats from Stage 4                 |
| `shards/*.tar`        | Shard files from Stage 6                 |

---

## Outputs

| Path                    | Description              |
|-------------------------|--------------------------|
| `manifests/<version>.json` | Versioned manifest    |

### Manifest JSON Schema

```json
{
  "version":    "v1",
  "created_at": "2025-04-25T10:30:00+00:00",
  "config": {
    "min_score":       4.5,
    "dedup_threshold": 0.95
  },
  "sources": {
    "video_dir":    "/path/to/raw_videos",
    "total_videos": 10000
  },
  "dedup": {
    "total":     9998,
    "kept":      9944,
    "removed":   54,
    "threshold": 0.95
  },
  "scoring": {
    "scored":               9998,
    "passing":              2993,
    "mean":                 4.261,
    "min_score_threshold":  4.5
  },
  "shards": {
    "count":       15,
    "total_bytes": 6117941248,
    "total_mb":    6117.9,
    "files": [
      {"name": "shard_00000.tar", "size_bytes": 409600000},
      ...
    ]
  }
}
```

---

## CLI

```bash
# Create
python manifest.py create \
  --version  v1 \
  --video_dir $SCRATCH_DIR/raw_videos \
  --dedup     $SCRATCH_DIR/dedup_results.json \
  --scores    $SCRATCH_DIR/scores.json \
  --shards    $SCRATCH_DIR/shards \
  --out       $SCRATCH_DIR/manifests/v1.json

# List all versions
python manifest.py list --manifest_dir $SCRATCH_DIR/manifests

# Compare versions
python manifest.py diff \
  --a $SCRATCH_DIR/manifests/v1.json \
  --b $SCRATCH_DIR/manifests/v2.json
```

### `create` arguments

| Argument            | Required | Description                         |
|---------------------|----------|-------------------------------------|
| `--version`         | yes      | Version string, e.g. `v1`           |
| `--video_dir`       | yes      | Source video directory              |
| `--dedup`           | yes      | Path to dedup_results.json          |
| `--scores`          | yes      | Path to scores.json                 |
| `--shards`          | yes      | Directory with shard .tar files     |
| `--out`             | yes      | Output manifest JSON path           |
| `--min_score`       | no       | Score threshold (default 4.5)       |
| `--dedup_threshold` | no       | Dedup threshold (default 0.95)      |

---

## v1 Run Metrics

| Metric          | Value              |
|-----------------|--------------------|
| Total videos    | 10,000             |
| After dedup     | 9,944              |
| Passing score   | 2,993              |
| Shards          | 15 files (5,833 MB)|

---

## Why Versioned Manifests

| Use Case                    | How manifests help                                      |
|-----------------------------|---------------------------------------------------------|
| Reproducibility             | Re-running with the same config produces identical data |
| Training attribution        | Log `manifest_version` in model card                   |
| Dataset evolution           | `diff` shows exactly what changed between v1 and v2    |
| Partial retraining          | Know exactly which shards to feed to the dataloader     |
| Compliance / audit          | Traceable record of all filtering thresholds applied    |

---

## wandb Metrics Logged

| Metric                      | Description                    |
|-----------------------------|--------------------------------|
| `manifest/version`          | Version string                 |
| `manifest/total_videos`     | Source video count             |
| `manifest/dedup_kept`       | Videos after dedup             |
| `manifest/dedup_removed`    | Videos removed by dedup        |
| `manifest/scoring_passing`  | Videos passing score filter    |
| `manifest/scoring_mean`     | Mean aesthetic score           |
| `manifest/n_shards`         | Number of shard files          |
| `manifest/total_mb`         | Total dataset size in MB       |

The manifest JSON is also uploaded as a wandb artifact via `wandb.save(str(out))`.
