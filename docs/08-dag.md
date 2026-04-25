# DAG Orchestration — dag.py

Full pipeline orchestration using **Prefect**. Wraps all 7 stages as Prefect tasks inside a single flow, with retry logic, task-level caching, and `--from_stage` support for resuming partial runs.

---

## What it does

- Defines one Prefect `@task` per pipeline stage, each of which shells out to the corresponding Python script via `subprocess.run`.
- Wraps all tasks in a `@flow` called `video-curation-pipeline`.
- Accepts `--from_stage` to skip stages that have already completed.
- Logs per-stage status (exit code, duration) to wandb.
- Initialises a parent wandb run that spans the entire pipeline.

---

## Flow Diagram

```
curation_pipeline()
    ├── task_ingest()    → subprocess: ingest.py
    ├── task_embed()     → subprocess: embed.py
    ├── task_dedup()     → subprocess: dedup.py
    ├── task_score()     → subprocess: score.py
    ├── task_caption()   → subprocess: caption.py
    ├── task_shard()     → subprocess: shard.py
    └── task_manifest()  → subprocess: manifest.py
```

Each task is gated by `stage_idx <= STAGES.index(from_stage)`, so stages before `--from_stage` are simply not called.

---

## CLI

```bash
# Full pipeline
python dag.py --version v1 --limit 10000 --num_gpus 4

# Resume from score stage (skip ingest, embed, dedup)
python dag.py --version v1 --limit 10000 --from_stage score --num_gpus 4

# Resume from shard stage
python dag.py --version v1 --limit 10000 --from_stage shard --num_gpus 4
```

| Argument              | Default    | Description                                       |
|-----------------------|------------|---------------------------------------------------|
| `--split`             | `val`      | Kinetics split for ingest                         |
| `--version`           | `v1`       | Dataset version string                            |
| `--limit`             | `500`      | Tar files to download in ingest (~50 clips each)  |
| `--workers`           | `8`        | Download threads for ingest                       |
| `--frames_per_video`  | `8`        | Frames sampled per video (embed, score, caption)  |
| `--num_gpus`          | `1`        | GPUs for Ray-based stages                         |
| `--dedup_threshold`   | `0.95`     | Cosine similarity threshold for dedup             |
| `--min_score`         | `4.5`      | Aesthetic score filter threshold                  |
| `--shard_size`        | `200`      | Samples per WebDataset shard                      |
| `--from_stage`        | `ingest`   | Resume from this stage (skip earlier ones)        |

Valid `--from_stage` values: `ingest`, `embed`, `dedup`, `score`, `caption`, `shard`, `manifest`.

---

## Prefect Task Configuration

| Task         | Retries | Cache                        | Retry delay |
|--------------|---------|------------------------------|-------------|
| `task_ingest`| 2       | `task_input_hash`, 24h TTL   | 30s         |
| `task_embed` | 1       | None                         | —           |
| `task_dedup` | 1       | None                         | —           |
| `task_score` | 1       | None                         | —           |
| `task_caption`| 1      | None                         | —           |
| `task_shard` | 1       | None                         | —           |
| `task_manifest`| 0     | None                         | —           |

Ingest has a 24h task cache because S3 downloads are expensive and idempotent — if ingest completes, subsequent Prefect runs with the same args will reuse the result.

Embed, score, and caption have their own per-video caches (`.npz`, `.score.json`, `.caption.json`), so Prefect-level task caching is not needed — the scripts handle their own idempotency internally.

---

## `_paths()` — Scratch Directory Resolution

All stage scripts use `$SCRATCH_DIR` as the root storage directory. `dag.py` resolves this once:

```python
scratch = Path(os.environ.get("SCRATCH_DIR") or 
               os.path.expanduser("~/scratch/video-curation"))
```

On the NTU HPC cluster, `SCRATCH_DIR` is set in the PBS job script. Locally it defaults to `~/scratch/video-curation`.

---

## Why Prefect over Airflow

| Dimension        | Prefect (used here)                        | Airflow                             |
|------------------|--------------------------------------------|-------------------------------------|
| Setup            | Zero-config local execution                | Requires Postgres + web server      |
| Portability      | Runs as a plain Python script              | Needs a scheduler daemon            |
| Cloud migration  | Easy upgrade to Prefect Cloud              | Managed Airflow (MWAA/GCP) required |
| DAG concept      | Identical: flows, tasks, retries           | Identical: DAGs, operators, retries |
| Interview optics | Demonstrates orchestrator knowledge        | Also valid, higher ops overhead     |

---

## wandb Integration

`dag.py` initialises a parent wandb run for the full pipeline:

```python
wandb.init(
    project="video-curation",
    name=f"{version}-{from_stage}",
    config={...all pipeline parameters...}
)
```

Each `_run()` call logs per-stage `{stage}/status` (1=success, 0=failure) and `{stage}/duration_s`.

---

## Running via PBS (Cluster)

The standard submission path is `qsub run_curation.pbs`, which sets environment variables and calls `python dag.py` with production parameters:

```bash
python dag.py \
    --split            val \
    --version          v1 \
    --limit            10000 \
    --frames_per_video 8 \
    --num_gpus         4 \
    --dedup_threshold  0.95 \
    --min_score        4.5 \
    --shard_size       200 \
    --from_stage       ingest
```
