# Stage 2 — embed.py · CLIP Embeddings

Extracts semantic embeddings for every video using CLIP ViT-B/32. Samples 8 frames per video via PyAV (CPU decode), runs them through CLIP on GPU, and stores mean + per-frame embeddings as `.npz` files.

---

## What it does

1. Scans `raw_videos/` for all `.mp4` files.
2. Spawns `num_gpus × 4` Ray actors, each holding a loaded CLIP model in GPU memory.
3. Round-robins videos across workers. Each worker:
   - Checks if the `.npz` already exists (cache hit → skip).
   - Decodes up to 64 frames via PyAV (CPU), samples down to 8 evenly spaced frames.
   - Runs the 8 frames through `CLIPModel.vision_model` → `visual_projection` → L2 normalise.
   - Writes `mean_embedding [512]` and `frame_embeddings [8, 512]` to `<stem>.npz`.
4. Logs progress to stdout and wandb.

---

## Inputs

| Path            | Description                      |
|-----------------|----------------------------------|
| `raw_videos/**/*.mp4` | All video clips from Stage 1 |

---

## Outputs

| Path                         | Description                                               |
|------------------------------|-----------------------------------------------------------|
| `embeddings/<stem>.npz`      | `mean_embedding [512]` + `frame_embeddings [8, 512]`      |

The `.npz` also stores `video_path` as a string array for dedup and stats.

---

## CLI

```bash
python embed.py \
  --video_dir $SCRATCH_DIR/raw_videos \
  --out_dir   $SCRATCH_DIR/embeddings \
  --num_gpus  4
```

| Argument              | Default            | Description                     |
|-----------------------|--------------------|---------------------------------|
| `--video_dir`         | required           | Directory with raw MP4 files    |
| `--out_dir`           | `data/embeddings`  | Output directory for .npz files |
| `--frames_per_video`  | `8`                | Frames sampled per video        |
| `--num_gpus`          | `1`                | GPUs to use (16 actors for 4)   |
| `--ray_address`       | `None`             | Existing Ray cluster address    |

---

## Model

**CLIP ViT-B/32** (`openai/clip-vit-base-patch32`)

The inference path calls sub-modules directly rather than `CLIPModel.get_image_features()`:

```python
vision_out = self.model.vision_model(pixel_values=inputs["pixel_values"])
feats = self.model.visual_projection(vision_out.pooler_output)
feats = F.normalize(feats, dim=-1)
```

This was necessary because `get_image_features()` returned a wrong type (`BaseModelOutputWithPooling` without `.norm`) on the installed transformers version (Bug 1 from the bug log).

---

## Concurrency Model

```
4 GPUs × 4 actors/GPU = 16 concurrent EmbedWorker actors
Each actor: num_gpus=0.25 (Ray resource fraction)
```

Each `EmbedWorker` loads CLIP once in `__init__` and serves many videos sequentially. This avoids reloading ~400 MB of weights per video.

---

## Decode Path — Why CPU (PyAV)

All Ray actors use `decode_video_actor()` from `decode.py`, which uses PyAV (CPU ffmpeg). **GPU decode (PyNvVideoCodec) is explicitly not used for actors.**

With `num_gpus=0.25`, four actors share one physical GPU. Every PyNvVideoCodec decoder creates its own CUDA context on the shared GPU via NVDEC. These contexts race, and DLPack pointers become invalid when any actor's context is destroyed, causing `CUDA_ERROR_CONTEXT_IS_DESTROYED` crashes.

CPU decode costs only a few ms per video vs ~50–100ms for CLIP inference — it is never the bottleneck. See `decode.py` documentation for the full bug history.

---

## v1 Run Metrics

| Metric              | Value  |
|---------------------|--------|
| Videos processed    | 10,000 |
| Cached (rerun)      | 9,998  |
| Failed              | 2      |
| Elapsed (cached)    | 22s    |

---

## wandb Metrics Logged

| Metric              | Description                   |
|---------------------|-------------------------------|
| `embed/ok`          | Successfully embedded videos  |
| `embed/cached`      | Skipped (npz already exists)  |
| `embed/failed`      | Decode or inference failures  |
| `embed/completed`   | Total processed so far        |
| `embed/total`       | Total videos to process       |
| `embed/elapsed_s`   | Wall time                     |

---

## Why ViT-B/32 not ViT-L/14

- 4× faster inference (B/32 vs L/14)
- B/32 embeds into the same CLIP semantic space — cosine similarity dedup works correctly
- ViT-L/14 is reserved for the aesthetic scorer (Stage 4), where the MLP was specifically trained on L/14 features
