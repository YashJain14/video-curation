# Documentation Index

| File | Contents |
|------|----------|
| [00-overview.md](00-overview.md) | Pipeline DAG, run results, GPU concurrency design, directory layout |
| [01-ingest.md](01-ingest.md) | Stage 1 — S3 download, tar extraction, label assignment |
| [02-embed.md](02-embed.md) | Stage 2 — CLIP ViT-B/32 embeddings, Ray concurrency, decode path |
| [03-dedup.md](03-dedup.md) | Stage 3 — MD5 exact dedup + FAISS near-dedup, Union-Find, MD5 cache |
| [04-score.md](04-score.md) | Stage 4 — LAION aesthetic scorer, CLIP ViT-L/14, filter threshold |
| [05-motion.md](05-motion.md) | Stage 5 — Optical flow + SSIM motion quality, threshold calibration |
| [05-caption.md](05-caption.md) | Stage 6 — Qwen3-VL-8B captioning, aesthetic+motion pre-filter |
| [06-shard.md](06-shard.md) | Stage 9 — WebDataset .tar shards, three-gate filter, latent embedding |
| [07-caption-quality.md](07-caption-quality.md) | Stage 7 — CLIP text-image alignment, caption quality scoring |
| [07-manifest.md](07-manifest.md) | Stage 10 — Versioned JSON manifests, create/list/diff, wandb logging |
| [08-dag.md](08-dag.md) | Prefect orchestration, --from_stage resumption, --no_v2 flag |
| [08-encode-latents.md](08-encode-latents.md) | Stage 8 — SD-VAE latent pre-encoding, batch encode, quality pre-filter |
| [09-decode.md](09-decode.md) | decode.py — CPU vs GPU paths, CUDA context bug history |
| [10-filter.md](10-filter.md) | filter.py — CLIP text-query semantic filtering (optional) |
| [11-stats.md](11-stats.md) | stats.py — dataset composition charts, PCA, label balance |
| [12-profiling.md](12-profiling.md) | profile_run.py — torchscope GPU profiling for the embed stage |
| [13-setup.md](13-setup.md) | setup_env.sh, prefetch_models.py, PBS job files, offline mode |
