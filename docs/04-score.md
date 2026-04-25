# Stage 4 — score.py · Aesthetic Scoring

Scores every video using the LAION aesthetic predictor — a small MLP trained on human aesthetic ratings from LAION-5B. Operates on CLIP ViT-L/14 embeddings. Scores are used downstream to filter low-quality clips before expensive VLM captioning.

---

## What it does

For each video:
1. Decodes frames via PyAV (CPU), samples down to `frames_per_video` evenly spaced frames.
2. Extracts CLIP ViT-L/14 embeddings (768-dim) via `clip.vision_model` → `clip.visual_projection`.
3. Passes embeddings through the LAION aesthetic MLP → scalar score per frame in [0, 10].
4. Computes `mean_score` across frames.
5. Writes per-video result to `score_cache/<stem>.score.json`.
6. Assembles all results into `scores.json`.

---

## Inputs

| Path                  | Description                      |
|-----------------------|----------------------------------|
| `raw_videos/**/*.mp4` | All video clips from Stage 1     |

---

## Outputs

| Path                             | Description                                      |
|----------------------------------|--------------------------------------------------|
| `scores.json`                    | List of `{path, mean_score, frame_scores, status}` |
| `score_cache/<stem>.score.json`  | Per-video cache — avoids re-scoring on reruns   |

### scores.json entry schema

```json
{
  "path":         "/path/to/video.mp4",
  "status":       "ok",
  "mean_score":   4.73,
  "frame_scores": [4.1, 4.9, 4.8, 4.6, 4.7, 4.8, 4.9, 4.5],
  "time_s":       0.08
}
```

---

## CLI

```bash
python score.py \
  --video_dir $SCRATCH_DIR/raw_videos \
  --out       $SCRATCH_DIR/scores.json \
  --num_gpus  4
```

| Argument              | Default           | Description                      |
|-----------------------|-------------------|----------------------------------|
| `--video_dir`         | required          | Directory with raw MP4 files     |
| `--out`               | `data/scores.json`| Output JSON path                 |
| `--frames_per_video`  | `8`               | Frames to sample per video       |
| `--min_score`         | `4.5`             | Threshold for pass/fail logging  |
| `--num_gpus`          | `1`               | GPUs to use (16 actors for 4)    |
| `--ray_address`       | `None`            | Existing Ray cluster address     |

---

## Models

### CLIP ViT-L/14 (`openai/clip-vit-large-patch14`)

Backbone for feature extraction. Outputs 768-dim embeddings.

**Why L/14 not B/32:** The LAION aesthetic MLP was trained specifically on ViT-L/14 embeddings. Using the wrong backbone produces meaningless scores — the MLP weights are tightly coupled to the L/14 feature space.

### LAION Aesthetic MLP (`camenduru/improved-aesthetic-predictor`)

```
Linear(768, 1024) → Dropout(0.2)
→ Linear(1024, 128) → Dropout(0.2)
→ Linear(128, 64)   → Dropout(0.1)
→ Linear(64, 16)
→ Linear(16, 1)
→ scalar score ∈ [0, 10]
```

Weights file: `sac+logos+ava1-l14-linearMSE.pth`

**Correct HuggingFace repo:** `camenduru/improved-aesthetic-predictor`. The originally-referenced `shunk031/improved-aesthetic-predictor` returns 404. The original author's repo `christophschuhmann/improved-aesthetic-predictor` also had issues at time of writing.

---

## Concurrency Model

```
4 GPUs × 4 actors/GPU = 16 concurrent ScoreWorker actors
Each actor: num_gpus=0.25
Each actor loads CLIP L/14 + AestheticMLP once in __init__
```

Weights are resolved via `hf_hub_download` on the driver before Ray starts to avoid HuggingFace API calls inside offline workers.

---

## Score Interpretation

| Range    | Meaning                                      |
|----------|----------------------------------------------|
| 0–3      | Very low quality (blurry, dark, noisy)       |
| 3–4.5    | Below threshold — filtered out               |
| 4.5–5.5  | Passable — goes to captioning                |
| 5.5–7    | Good quality                                 |
| 7–10     | High quality (rare in Kinetics-400)          |

The threshold used in this pipeline is **4.5**. About 29.9% of Kinetics-400 val passes this threshold. This is expected — Kinetics-400 is an action recognition dataset sourced from YouTube, not curated for aesthetic quality.

---

## v1 Run Metrics

| Metric            | Value             |
|-------------------|-------------------|
| Videos scored     | 9,998             |
| Failed            | 2                 |
| Mean score        | 4.26              |
| Passing (≥ 4.5)   | 2,993 (29.9%)     |
| Elapsed (cached)  | 23s               |

---

## wandb Metrics Logged

| Metric             | Description                    |
|--------------------|--------------------------------|
| `score/ok`         | Successfully scored videos     |
| `score/failed`     | Decode or inference failures   |
| `score/completed`  | Total processed so far         |
| `score/total`      | Total videos to score          |
| `score/elapsed_s`  | Wall time                      |

---

## Why Score Before Caption

Aesthetic scoring runs at ~0.1s/video. VLM captioning (Qwen3-VL-8B) runs at ~3s/video — 30× slower. Filtering 70% of videos before captioning saves the equivalent of ~2 hours of GPU time on 10k videos.

This is the core principle of **filter-first, caption-second** pipelines.
