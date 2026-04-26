# Stage 5 — motion.py · Motion Quality Scoring

CPU-only Ray stage that scores temporal quality for every video using dense optical flow and structural similarity. Runs on all videos before captioning so the expensive VLM step only processes clips with good temporal structure.

---

## Why Motion Quality

Aesthetic score measures individual frame quality — a sharp, well-lit frame scores high even if the clip is frozen, shaky, or contains a scene cut. Video diffusion models learn temporal consistency from training data; clips with erratic motion or discontinuities actively harm the model by teaching it that frames within a clip can be unrelated.

Three problems aesthetic score cannot detect:
- **Frozen/near-static clips** — high aesthetic score (sharp frame), useless for motion learning
- **Shaky handheld footage** — high optical flow variance, but individual frames look fine
- **Scene cuts inside the clip** — SSIM drops sharply at the cut; optical flow spikes

---

## Signals

**`flow_magnitude`** — mean Farneback dense optical flow magnitude across sampled frame pairs (pixels/frame). Low = frozen; very high = chaotic.

**`flow_variance`** — variance of per-pair flow magnitudes. High = unsteady camera shake.

**`ssim_mean`** — mean SSIM between consecutive sampled frames. Low = scene cut, heavy blur, or flicker. High = temporally coherent.

### Composite `motion_score [0–10]`

```python
coherence   = ssim_mean * 5.0                         # 0–5: temporal smoothness
flow_reward = 3.0 * exp(-((flow_mag - 5.0)**2) / 50)  # 0–3: peaks at moderate motion ~5–8 px
stability   = 2.0 * exp(-flow_var / 10.0)             # 0–2: penalises shaky footage
motion_score = clip(coherence + flow_reward + stability, 0, 10)
```

Design targets:
- Smooth, moderately moving clip → 8–9
- Slow pan / nearly static → 5–6 (ssim high, flow low)
- Shaky handheld → 3–5 (flow_var high)
- Scene cut inside clip → 1–3 (ssim drops sharply)

**8 frame pairs sampled** per video (evenly spaced via PyAV).

---

## Inputs / Outputs

| | Path |
|--|------|
| Input | `raw_videos/**/*.mp4` |
| Output | `motion_scores.json` — `[{path, flow_magnitude, flow_variance, ssim_mean, motion_score, status}]` |
| Cache | `motion_cache/<stem>.motion.json` — per-video, skipped on rerun |

---

## CLI

```bash
python motion.py \
  --video_dir   $SCRATCH_DIR/raw_videos \
  --out         $SCRATCH_DIR/motion_scores.json \
  --num_workers 16
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--video_dir` | required | Source video directory |
| `--out` | `data/motion_scores.json` | Output path |
| `--num_workers` | `16` | Number of CPU Ray actors (2 CPUs each) |
| `--min_motion` | `5.0` | Threshold for summary stats only (filtering applied in caption.py / shard.py) |

---

## Concurrency

CPU-only stage. `@ray.remote(num_cpus=2)` — each worker gets 2 CPU cores for OpenCV's Farneback implementation. 16 workers × 2 CPUs = 32 cores, saturating a typical HPC node.

No GPU is required or used.

---

## Threshold Calibration

Threshold was calibrated from the actual Kinetics-400 distribution:

| Percentile | Score |
|-----------|-------|
| p25 | 5.11 |
| p50 (median) | 6.61 |
| Mean | 6.42 |

**Threshold = 5.0** (p25) removes the bottom quartile of temporal quality. This eliminates the worst frozen/shaky clips while retaining 76.7% of aesthetically-passing videos for captioning.

An earlier attempt used threshold = 3.0 (placeholder) which passed 95.6% — meaningless filtering. The p25 value was determined by measuring the real distribution.

---

## Run Metrics

| Metric | Value |
|--------|-------|
| Videos scored | 9,998 |
| Mean motion score | 6.42 |
| Median | 6.61 |
| Mean flow magnitude | 7.07 px |
| Mean SSIM | 0.588 |
| Passing (≥ 5.0) | 7,665 (76.7% of all; filters 594 from aesthetic-passing set) |
| Wall time | ~4 min |

---

## wandb Metrics Logged

| Metric | Description |
|--------|-------------|
| `motion/score_mean` | Mean motion score across all videos |
| `motion/score_median` | Median motion score |
| `motion/flow_mag_mean` | Mean optical flow magnitude |
| `motion/ssim_mean` | Mean SSIM |
| `motion/passing` | Count passing threshold |
| `motion/passing_pct` | Percentage passing |
| `motion/score_hist` | Histogram of motion scores |
| `motion/flow_hist` | Histogram of flow magnitudes |
| `motion/ssim_hist` | Histogram of SSIM values |
