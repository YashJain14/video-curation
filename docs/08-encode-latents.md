# Stage 8 — encode_latents.py · VAE Latent Pre-encoding

GPU Ray stage that pre-encodes video frames into VAE latent tensors. Training DataLoaders read these tensors directly, eliminating repeated VAE encoding during training.

---

## Why Pre-encode Latents

Video DiTs (CogVideoX, Open-Sora, Wan, HunyuanVideo) operate in latent space. During training, every batch re-encodes each video through the VAE encoder before the diffusion forward pass. On a large dataset trained for many epochs, this cost compounds:

- **Without pre-encoding:** VAE forward pass runs on every sample, every step, every epoch
- **With pre-encoding:** VAE runs once at data preparation time; training reads float16 tensors from disk

Typical saving: **15–30% of training GPU time**, depending on resolution and sequence length. Higher at lower resolutions (smaller model overhead) and longer sequences (more frames to encode).

---

## VAE Used

`stabilityai/sd-vae-ft-mse` — the Stable Diffusion VAE:
- 4-channel latent space
- 8× spatial downsampling (256px → 32px latent)
- Fine-tuned on mean squared error for better reconstruction quality

Output shape: `[4, T, H//8, W//8]` float16 — e.g. `[4, 16, 32, 32]` for 16 frames at 256px.

> **In production:** Swap for the VAE used by the target training codebase. CogVideoX uses a 3D causal VAE; HunyuanVideo uses its own. The spatial 2D VAE here is appropriate for image DiT backbones adapted to video.

---

## Quality Pre-filtering

Before encoding, the stage intersects all three quality gates:
- Aesthetic score ≥ `min_score` (from `scores.json`)
- Motion score ≥ `min_motion` (from `motion_scores.json`)
- Caption quality ≥ `min_quality` (from `caption_quality.json`)

Only videos passing all three gates get latents encoded. This avoids spending VAE compute on clips that will be dropped at shard time.

From the run: **2,161 / 10,000 videos** passed all three gates and received latents.

---

## Batch Encoding

All T frames are encoded in a single VAE forward pass:

```python
frames = _decode_frames_pyav(video_path, num_frames, resolution)  # [T, 3, H, W]
x = frames.to(device)
lat = vae.encode(x).latent_dist.mode()      # [T, 4, H//8, W//8]
latent_tensor = lat.permute(1, 0, 2, 3).cpu().half()  # [4, T, H//8, W//8]
```

Earlier version encoded one frame at a time in a loop — ~16× slower. Batch encode reduced encode_latents wall time significantly.

Frames are decoded via PyAV (CPU) and resized to `resolution × resolution` before GPU encoding.

---

## Inputs / Outputs

| | Path |
|--|------|
| Input | `raw_videos/**/*.mp4` |
| Input | `scores.json`, `motion_scores.json`, `caption_quality.json` (for pre-filtering) |
| Output | `latents/<stem>.pt` — float16 tensor `[4, T, H//8, W//8]` |
| Cache | Per-file `.pt` existence check — skipped if already encoded |

Latent files are embedded into WebDataset shards by `shard.py` as `<key>.latent.pt`.

---

## CLI

```bash
python encode_latents.py \
  --video_dir        $SCRATCH_DIR/raw_videos \
  --out_dir          $SCRATCH_DIR/latents \
  --scores           $SCRATCH_DIR/scores.json \
  --motion           $SCRATCH_DIR/motion_scores.json \
  --captions_quality $SCRATCH_DIR/caption_quality.json \
  --min_score        4.5 \
  --min_motion       5.0 \
  --min_quality      0.65 \
  --num_gpus         4 \
  --num_frames       16 \
  --resolution       256
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--video_dir` | required | Source video directory |
| `--out_dir` | `data/latents` | Output directory for .pt files |
| `--scores` | `None` | scores.json for pre-filtering |
| `--motion` | `None` | motion_scores.json for pre-filtering |
| `--captions_quality` | `None` | caption_quality.json for pre-filtering |
| `--min_score` | `4.5` | Aesthetic gate |
| `--min_motion` | `5.0` | Motion gate |
| `--min_quality` | `0.65` | Caption quality gate |
| `--num_gpus` | `1` | Number of GPUs for Ray actors |
| `--num_frames` | `16` | Frames per video to encode (temporal depth) |
| `--resolution` | `256` | Spatial resolution before VAE (VAE downsamples by 8×) |

---

## Concurrency

`@ray.remote(num_gpus=0.5)` — 2 actors per GPU, 8 concurrent actors across 4 GPUs. Each actor loads the SD-VAE once on init and processes videos in sequence.

---

## Run Metrics

| Metric | Value |
|--------|-------|
| Videos passing all gates | 2,161 |
| Successfully encoded | 2,161 (0 failures) |
| Latent shape | [4, 16, 32, 32] |
| Wall time | 41:42 (2,474s) |
| Throughput | ~52 videos/min across 8 actors |

Encode_latents is the pipeline bottleneck — ~41 min of the ~66 min total. This is a one-time cost; reruns use the cache.

---

## wandb Metrics Logged

| Metric | Description |
|--------|-------------|
| `latents/ok` | Successfully encoded |
| `latents/cached` | Cache hits |
| `latents/failed` | Failures |
| `latents/elapsed_s` | Wall time |
| `latents/total_files` | Total .pt files on disk |
| `latents/total_mb` | Total disk usage |
