# Stage 7 — caption_quality.py · Caption Quality Analysis

GPU Ray stage that verifies VLM-generated captions are semantically accurate before they enter training shards. Uses CLIP text-image alignment plus heuristic signals to score each caption.

---

## Why Caption Quality

Captions are the text conditioning signal for text-to-video diffusion models. A vague or hallucinated caption ("a person is doing something") produces a weak text-visual association and degrades the model's ability to follow text prompts at inference. Qwen3-VL-8B occasionally:
- Hallucinates subjects or actions not present in the video
- Produces generic non-specific descriptions
- Generates very short captions that carry insufficient conditioning signal

This stage scores every caption and applies a quality gate before latent encoding and sharding.

---

## Three Signals

### 1. `clip_alignment` (0–1)
Cosine similarity between the CLIP ViT-B/32 text embedding of the caption and the CLIP video embedding from `embed.py`. High alignment = the caption accurately describes what is visually present. Low alignment = hallucination or off-topic description.

**Reuses embed.py output** — only the caption text is encoded; no video re-processing.

Typical range for Kinetics-400: 0.15–0.40.

### 2. `caption_length` (tokens)
Number of whitespace-delimited tokens. Very short captions (<5 tokens) carry insufficient conditioning signal. Very long captions (>60 tokens) often contain repetition or hallucination.

### 3. `specificity_score` (0–1)
Ratio of non-stopword tokens. A caption like "a skateboarder executes a kickflip on a concrete half-pipe" scores high; "a person is doing something in a place" scores low. Correlates with useful conditioning signal.

Uses a minimal hardcoded stopword list to avoid NLTK dependency.

---

## Composite `quality_score [0–1]`

```python
align_norm = clip((clip_alignment - 0.10) / 0.30, 0, 1)  # normalise [0.10, 0.40] → [0, 1]

if length < 5:      len_score = length / 5.0 * 0.5
elif length <= 30:  len_score = 0.5 + (length - 5) / 50.0
else:               len_score = max(0, 1.0 - (length - 30) / 60.0)

quality_score = 0.50 * align_norm + 0.30 * specificity + 0.20 * len_score
```

Weights reflect diffusion conditioning priorities: alignment (50%) is most important — is the caption accurate? Specificity (30%) — is it descriptive? Length (20%) — is it in the useful range?

---

## Inputs / Outputs

| | Path |
|--|------|
| Input | `captions.json` from caption.py |
| Input | `embeddings/*.npz` from embed.py (video embeddings, no re-encoding) |
| Output | `caption_quality.json` — `[{path, caption, clip_alignment, caption_length, specificity_score, quality_score, status}]` |
| Cache | `caption_quality_cache/<stem>.cq.json` — per-video, skipped on rerun |

---

## CLI

```bash
python caption_quality.py \
  --captions  $SCRATCH_DIR/captions.json \
  --emb_dir   $SCRATCH_DIR/embeddings \
  --out       $SCRATCH_DIR/caption_quality.json \
  --num_gpus  4
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--captions` | required | captions.json from caption.py |
| `--emb_dir` | required | Embeddings directory from embed.py |
| `--out` | `data/caption_quality.json` | Output path |
| `--num_gpus` | `1` | Number of GPUs for Ray actors |
| `--min_quality` | `0.65` | Threshold for summary stats (filtering applied in shard.py) |
| `--min_alignment` | `0.20` | CLIP alignment threshold for summary stats |

---

## Concurrency

`@ray.remote(num_gpus=0.5)` — 2 actors per GPU, 8 concurrent actors across 4 GPUs. Each actor loads CLIP ViT-B/32 once and processes captions in sequence. Only the caption text is sent to GPU (text encoder) — video embeddings are read from disk.

---

## Threshold Calibration

Distribution from the actual run:

| Metric | Value |
|--------|-------|
| Mean CLIP alignment | 0.319 |
| Median CLIP alignment | 0.320 |
| Passing alignment ≥ 0.20 | 2,384 / 2,396 (99.5%) |
| Mean caption length | 27.0 tokens |
| Mean specificity | 0.622 |
| Mean quality score | 0.730 |
| **Passing quality ≥ 0.65** | **2,161 / 2,396 (90.2%)** |

**Threshold = 0.65** (~p50 of the quality distribution) removes the bottom half. An earlier placeholder of 0.3 passed 100% of captions — meaningless.

---

## Run Metrics

| Metric | Value |
|--------|-------|
| Captions scored | 2,396 |
| Passing (≥ 0.65) | 2,161 (90.2%) |
| Wall time | 2:02 |

---

## wandb Metrics Logged

| Metric | Description |
|--------|-------------|
| `cq/alignment_mean` | Mean CLIP text-image alignment |
| `cq/alignment_median` | Median CLIP alignment |
| `cq/length_mean` | Mean caption length (tokens) |
| `cq/specificity_mean` | Mean specificity score |
| `cq/quality_mean` | Mean composite quality score |
| `cq/passing_quality` | Count passing quality threshold |
| `cq/passing_alignment` | Count passing alignment threshold |
| `cq/alignment_hist` | Histogram |
| `cq/length_hist` | Histogram |
| `cq/quality_hist` | Histogram |
