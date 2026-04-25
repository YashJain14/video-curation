# GPU Profiling — profile_run.py

GPU profiling wrapper for the embed stage using **torchscope**. Re-runs embedding under `torchscope.RayProfiler` to measure real GPU utilisation across all 16 Ray workers. Proves that the `0.25 GPU/task` allocation actually loads all 4 GPUs efficiently rather than just reserving them.

---

## What it does

1. Optionally imports torchscope from a local repo path (no pip install needed).
2. Creates a named `_AggregatorActor` that collects GPU metrics from all workers.
3. Starts a `Profiler` on the driver to record per-GPU timeline.
4. Spawns 16 `EmbedWorkerProfiled` actors (same logic as embed.py, `num_gpus=0.25`).
5. Each actor resolves the named aggregator by name inside its own process and starts a `worker_profiler_context` — avoids the un-picklable actor handle problem.
6. Dispatches all videos, collecting results.
7. Stops profiling in each actor.
8. Generates two HTML reports:
   - `embed_cluster_profile.html` — per-worker + cluster summary
   - `embed_driver_profile.html` — driver-side GPU utilisation timeline

---

## Outputs

| Path                                     | Description                                |
|------------------------------------------|--------------------------------------------|
| `reports/embed_cluster_profile.html`     | Per-worker GPU util, cluster stats, imbalance |
| `reports/embed_driver_profile.html`      | Driver GPU utilisation timeline            |
| `reports/embed_gpu_profile.json`         | Raw GPU profile data (from Profiler)       |
| `embeddings/<stem>.npz`                  | Embedding files (same as embed.py)         |

---

## CLI

```bash
python profile_run.py \
  --video_dir    $SCRATCH_DIR/raw_videos \
  --out_dir      $SCRATCH_DIR/embeddings \
  --num_gpus     4 \
  --torchscope   ~/torchscope \
  --force        \
  --limit        500
```

| Argument              | Default          | Description                                         |
|-----------------------|------------------|-----------------------------------------------------|
| `--video_dir`         | required         | Directory with raw MP4 files                        |
| `--out_dir`           | `$SCRATCH_DIR/embeddings` | Embedding output directory              |
| `--report_dir`        | `$SCRATCH_DIR/reports` | HTML report output directory               |
| `--frames_per_video`  | `8`              | Frames per video for CLIP                           |
| `--num_gpus`          | `1`              | Number of GPUs (16 actors for 4)                    |
| `--torchscope`        | `None`           | Path to torchscope repo root                        |
| `--force`             | `False`          | Re-embed even if .npz exists (needed for profiling) |
| `--limit`             | `None`           | Only process first N videos                         |

**Use `--force` for profiling.** Without it, all videos are cache hits and no GPU work happens, so the profiler sees 0% utilisation.

**Use `--limit`** with `--force` to profile a representative subset without re-embedding all 10k videos.

---

## torchscope Setup

torchscope is used as a local checkout, not a pip package:

```bash
git clone https://github.com/YashJain14/torchscope.git ~/torchscope
```

Pass `--torchscope ~/torchscope` and the script adds the path to `sys.path` before importing. This avoids any versioning conflicts and works on cluster nodes without internet access.

---

## Why Named Actor Pattern

The `RayProfiler` aggregator actor can't be pickled and passed across Ray task boundaries to workers. Instead:

1. The driver creates the aggregator with a fixed name: `"torchscope_aggregator"`.
2. Each worker resolves the aggregator by name inside its own process: `ray.get_actor("torchscope_aggregator")`.
3. Workers construct a minimal `RayProfiler` stub pointing at the named actor handle.

This avoids the `CloudPickleError: can't pickle actor handle` failure that occurs when passing the actor handle directly in the `EmbedWorkerProfiled.remote(...)` constructor call.

---

## Cluster Reports

```bash
# After the job completes, download the HTML reports:
scp user@cluster:~/scratch/video-curation/reports/*.html .
```

The `embed_cluster_profile.html` report shows:
- Per-worker average GPU utilisation
- Bottleneck worker ID (highest utilisation)
- Cluster imbalance percentage
- Timeline view of when each worker was active

---

## PBS Submission

```bash
qsub run_profile.pbs
```

`run_profile.pbs` requests 4 GPUs with a 30-minute walltime and runs:
```bash
python profile_run.py \
  --video_dir  $SCRATCH_DIR/raw_videos \
  --out_dir    $SCRATCH_DIR/embeddings \
  --report_dir $SCRATCH_DIR/reports \
  --num_gpus   4 \
  --torchscope ~/torchscope \
  --force \
  --limit 500
```

---

## `EmbedWorkerProfiled` vs `EmbedWorker`

Both actors are functionally identical. The profiled version adds:

1. `__init__`: constructs a `worker_profiler_context` by resolving the named aggregator.
2. Wraps the profiling context's lifecycle — `__enter__` in `__init__`, `__exit__` via `stop_profiling()`.
3. `stop_profiling()` remote method so the driver can cleanly terminate profiling before collecting reports.

The `force` flag bypasses the `.npz` cache check, ensuring GPU work actually happens.
