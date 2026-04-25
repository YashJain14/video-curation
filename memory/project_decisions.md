---
name: video-curation pipeline design decisions
description: Key architectural and implementation decisions made during April 2026 pipeline development — model choices, filtering strategy, decode path, data flow
type: project
---

## Decision 1 — Aesthetic predictor HuggingFace repo

**Decision:** Use `camenduru/improved-aesthetic-predictor` for the LAION aesthetic MLP weights.

**Why:** The originally hardcoded repo `shunk031/improved-aesthetic-predictor` returned 404. `christophschuhmann/improved-aesthetic-predictor` (the original author) also had issues. `camenduru` mirrors the correct weights including `sac+logos+ava1-l14-linearMSE.pth`.

**How to apply:** If the aesthetic predictor weights need to be re-fetched, use `camenduru/improved-aesthetic-predictor`.

---

## Decision 2 — Switch caption model from Qwen2.5-VL-7B to Qwen3-VL-8B

**Decision:** Upgraded captioning from `Qwen/Qwen2.5-VL-7B-Instruct` to `Qwen/Qwen3-VL-8B-Instruct`.

**Why:** Qwen3-VL-8B is the newer generation with better video understanding. Slightly larger (8B vs 7B) but same A100 memory footprint in bf16.

**API differences from Qwen2.5-VL:**
- Class: `Qwen3VLForConditionalGeneration` (not `Qwen2_5_VLForConditionalGeneration`)
- Loading: `dtype="auto"` instead of `torch_dtype=torch.bfloat16`
- Inference: `apply_chat_template` with `tokenize=False` + separate `process_vision_info` call (not `tokenize=True, return_dict=True`)
- `process_vision_info` returns 3-tuple `(image_inputs, video_inputs, video_kwargs)` when `return_video_kwargs=True`

**How to apply:** Always check qwen-vl-utils version when upgrading Qwen models — API changes between minor versions.

---

## Decision 3 — Native video input to Qwen3-VL (no manual frame extraction)

**Decision:** Pass raw video file path directly to Qwen3-VL via `{"type": "video", "video": path, "max_frames": N}` instead of manually decoding frames and passing as images.

**Why:** Eliminates the entire decode → sample → PIL conversion pipeline (decode_video_actor, _sample_frames_pil, numpy, PIL). Simpler code, fewer failure points, model handles frame sampling internally tuned to its own attention patterns.

**How to apply:** For any VLM that supports native video input, prefer that over manual frame extraction.

---

## Decision 4 — Caption only videos passing score filter

**Decision:** Added `--scores` and `--min_score` args to `caption.py` so it only captions videos with `mean_score >= 4.5`.

**Why:** Only 29.9% of 10k videos pass the aesthetic score threshold. Captioning all 10k videos wastes ~70% of Qwen3-VL compute on videos that get thrown away in shard stage. Score filtering before captioning gives ~3x speedup.

**Order matters:** Score → filter → caption → shard (not score → caption → filter).

**How to apply:** Always apply cheap filters (aesthetic score, dedup) before expensive VLM inference stages.

---

## Decision 5 — Skip videos with empty captions in shard stage

**Decision:** `shard.py` skips any video where `caption == ""`.

**Why:** Previously, videos that failed captioning or weren't captioned would be written to shards with empty `.txt` files. Training on empty captions is harmful. Now shards only contain videos with both `score >= min_score` AND a non-empty caption.

**How to apply:** Always validate all required fields exist before writing to training shards.

---

## Decision 6 — qwen-vl-utils fps list bug workaround

**Symptom:** `process_vision_info` with `return_video_kwargs=True` and decord backend returns `fps` as a list `[0.799...]` instead of a scalar, causing `Validation error for field 'fps'` in the processor.

**Fix:** Flatten fps before passing to processor:
```python
if "fps" in video_kwargs and isinstance(video_kwargs["fps"], list):
    video_kwargs["fps"] = video_kwargs["fps"][0] if video_kwargs["fps"] else 1.0
```

**Why:** Bug in qwen-vl-utils 0.0.14 (Qwen2-VL era) when used with decord backend. The `return_video_metadata=True` flag makes it worse — avoid that parameter entirely.

**How to apply:** If upgrading qwen-vl-utils, re-test whether fps is still returned as a list.

---

## Decision 7 — caption.py decode path: decode_video_actor not decode_video

**Decision:** `caption.py` uses `decode_video_actor` (PyAV CPU) not `decode_video` (ThreadedDecoder GPU).

**Why:** Same reason as embed/score stages — ThreadedDecoder creates CUDA contexts that race with Qwen3-VL's own CUDA context on the shared GPU, causing SIGABRT. With `num_gpus=1` per CaptionWorker (full GPU), GPU decode would technically be safe, but PyAV CPU decode costs only a few ms vs hundreds of ms for VLM inference — not worth the risk.

**After switching to native video input:** `decode_video_actor` is no longer used in caption.py at all — Qwen3-VL reads the video file directly via qwen_vl_utils/decord.

---

## Pipeline final stats (v1, April 2026)

| Stage | Result |
|---|---|
| Ingested | 10,000 videos (Kinetics-400 val, 10 tar files) |
| After dedup | 9,944 kept, 56 removed |
| Score ≥ 4.5 | 2,993 / 9,998 scored (29.9%) |
| Captioned | ~2,993 videos (Qwen3-VL-8B) |
| Shards | 15 × ~390 MB = 5.8 GB WebDataset |
| Total runtime | ~1.5h on 4× A100-SXM4-40GB |

**Score distribution note:** Kinetics-400 is an action recognition dataset, not an aesthetic dataset — low scores expected. Mean aesthetic score was 4.26 across all videos.
