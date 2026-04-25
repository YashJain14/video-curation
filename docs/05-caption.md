# Stage 5 — caption.py · VLM Captioning

Generates natural language captions for each video that passed the aesthetic score filter using **Qwen3-VL-8B-Instruct**. Passes raw video files directly to the model — no manual frame extraction pipeline.

---

## What it does

1. Loads `scores.json` and filters to videos with `mean_score ≥ min_score` (default 4.5). This eliminates ~70% of videos before running any inference.
2. Spawns one `CaptionWorker` Ray actor per GPU (1 GPU each — the model needs the full 40 GB).
3. Each worker:
   - Checks `caption_cache/<stem>.caption.json` — returns cached result if present.
   - Constructs a chat message with the video path and caption prompt.
   - Runs `process_vision_info` to decode frames internally via decord.
   - Applies a bug fix for the `fps` list issue in `qwen-vl-utils==0.0.14`.
   - Generates up to 128 new tokens with `model.generate`.
   - Trims the prompt tokens from the output, decodes to a string.
   - Saves caption to cache and returns the result.
4. Assembles all results into `captions.json`.

---

## Inputs

| Path                  | Description                                     |
|-----------------------|-------------------------------------------------|
| `raw_videos/**/*.mp4` | All video clips from Stage 1                    |
| `scores.json`         | Scores from Stage 4 (used for pre-filtering)    |

---

## Outputs

| Path                               | Description                                    |
|------------------------------------|------------------------------------------------|
| `captions.json`                    | List of `{path, caption, status, time_s}`      |
| `caption_cache/<stem>.caption.json`| Per-video cache — avoids re-captioning reruns  |

### captions.json entry schema

```json
{
  "path":    "/path/to/video.mp4",
  "status":  "ok",
  "caption": "A person performs a backflip on a trampoline in a gymnasium.",
  "time_s":  2.8
}
```

---

## CLI

```bash
python caption.py \
  --video_dir $SCRATCH_DIR/raw_videos \
  --out       $SCRATCH_DIR/captions.json \
  --scores    $SCRATCH_DIR/scores.json \
  --min_score 4.5 \
  --num_gpus  4
```

| Argument              | Default              | Description                               |
|-----------------------|----------------------|-------------------------------------------|
| `--video_dir`         | required             | Directory with raw MP4 files              |
| `--out`               | `data/captions.json` | Output JSON path                          |
| `--frames_per_video`  | `16`                 | Max frames for the model to sample        |
| `--num_gpus`          | `1`                  | One worker actor per GPU                  |
| `--scores`            | `None`               | Path to scores.json for pre-filtering     |
| `--min_score`         | `4.5`                | Skip videos below this score threshold    |
| `--ray_address`       | `None`               | Existing Ray cluster address              |

---

## Model: Qwen3-VL-8B-Instruct

**HuggingFace ID:** `Qwen/Qwen3-VL-8B-Instruct`

Key settings:
```python
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID, dtype="auto", device_map="auto"
)
```

- `dtype="auto"` selects bf16 automatically on A100 (not `torch_dtype=torch.bfloat16` — Qwen3-VL API change from Qwen2.5-VL)
- `device_map="auto"` places the model on the available GPU
- Model needs the full 40 GB of an A100 in bf16 — `num_gpus=1.0` per actor

### Caption Prompt

```
"You are a video captioning assistant. Given a short video clip, write a single concise 
sentence describing the main action, subject, and setting. Be specific about motion and 
appearance."
```

### Inference settings

| Parameter        | Value        |
|------------------|--------------|
| `max_frames`     | 8–16         |
| `max_pixels`     | 128×32×32    |
| `max_new_tokens` | 128          |

---

## Native Video Input

Qwen3-VL accepts a video file path directly in the message payload:

```python
{
    "type": "video",
    "video": video_path,
    "max_pixels": 128 * 32 * 32,
    "max_frames": frames_per_video,
}
```

The model handles frame sampling internally via decord. This eliminates the manual decode → frame sample → PIL pipeline required by older vision models.

---

## Known Bugs and Fixes

### Qwen3-VL vs Qwen2.5-VL API Differences

| Aspect          | Qwen2.5-VL                                  | Qwen3-VL                                    |
|-----------------|---------------------------------------------|---------------------------------------------|
| Class           | `Qwen2_5_VLForConditionalGeneration`        | `Qwen3VLForConditionalGeneration`           |
| dtype           | `torch_dtype=torch.bfloat16`               | `dtype="auto"`                              |
| Inference       | Direct tokenizer call                       | `apply_chat_template` + `process_vision_info` |
| `process_vision_info` | returns `(image_inputs, video_inputs)` | returns `(image_inputs, video_inputs, video_kwargs)` when `return_video_kwargs=True` |

### qwen-vl-utils fps List Bug (version 0.0.14)

`process_vision_info` with `return_video_kwargs=True` and the decord backend returns `fps` as a list `[0.799...]` instead of a scalar, causing a validation error inside the processor.

**Fix applied in `caption.py`:**
```python
if "fps" in video_kwargs and isinstance(video_kwargs["fps"], list):
    video_kwargs["fps"] = video_kwargs["fps"][0] if video_kwargs["fps"] else 1.0
```

**Do not** pass `return_video_metadata=True` — it worsens the issue.

---

## Concurrency Model

```
4 GPUs × 1 actor/GPU = 4 concurrent CaptionWorker actors
Each actor: num_gpus=1.0
```

Qwen3-VL-8B in bf16 occupies ~16 GB of the 40 GB A100. `device_map="auto"` fills the entire GPU. Only one instance per GPU is safe.

---

## v1 Run Metrics

| Metric           | Value                       |
|------------------|-----------------------------|
| Videos captioned | 2,993                       |
| Successful       | 2,993 (100%)                |
| Failed           | 0                           |
| Elapsed          | 187s (~3.1 min)             |
| Throughput       | ~16 videos/min across 4 GPUs|

---

## wandb Metrics Logged

| Metric               | Description                    |
|----------------------|--------------------------------|
| `caption/ok`         | Successfully captioned videos  |
| `caption/failed`     | Inference failures             |
| `caption/completed`  | Total processed so far         |
| `caption/total`      | Total videos to caption        |
| `caption/elapsed_s`  | Wall time                      |
