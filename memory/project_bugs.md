---
name: video-curation pipeline bugs
description: Two bugs found and fixed in the embed stage during April 2026 debugging session
type: project
---

## Bug 1 — `CLIPModel.get_image_features()` returns wrong type (transformers version mismatch)

**Symptom:** 100% of videos silently failed with `failed: 'BaseModelOutputWithPooling' object has no attribute 'norm'`. No crash, no stack trace in the output — the exception was caught by the `try/except` and returned as a status string that was never printed.

**Root cause:** The installed `transformers` version has a bug where `CLIPModel.get_image_features()` and `get_text_features()` return the raw encoder `BaseModelOutputWithPooling` object instead of the projected embedding tensor.

**Fix:** Bypass those methods entirely and call the submodules directly:
- `embed.py`: `vision_model(pixel_values=...) → visual_projection(pooler_output)`
- `score.py`: same pattern for the LAION aesthetic scorer's CLIP backbone
- `filter.py`: `text_model(input_ids=..., attention_mask=...) → text_projection(pooler_output)`

**Why:** Going through `vision_model` → `visual_projection` directly is version-stable and produces identical results to what `get_image_features` is supposed to return.

**How to apply:** Any future code using `CLIPModel` on this cluster should use the submodule pattern, not `get_image_features` / `get_text_features`.

---

## Bug 2 — `ThreadedDecoder.end()` intermittent SIGSEGV (PyNvVideoCodec 2.1 bug)

**Symptom:** After processing ~750/10000 videos successfully, one EmbedWorker crashes with SIGSEGV inside `ThreadedDecoder.end()`. Ray kills the actor, Prefect retries the whole embed stage. The retry succeeds because completed videos are cached (`.npz` already written).

**Root cause:** PyNvVideoCodec 2.1 `ThreadedDecoder.end()` segfaults intermittently when called while background decode threads are still active. It is a bug in PyNvVideoCodec's C++ layer, not in the Python code.

**Stack trace:**
```
File "ThreadedDecoder.py", line 92 in end
File "decode.py", line 36 in decode_gpu
```

**Root cause (fully diagnosed):** `CUDA_ERROR_CONTEXT_IS_DESTROYED` in `ExternalBuffer.cpp` / `NvDecoder.cpp`. `ThreadedDecoder` internally creates a CUDA context via NVDEC (`cuvidCreateDecoder`) and runs a persistent background decode thread holding that context. With Ray's 0.25 GPU fractional allocation, 4 actors share one physical GPU and each independently instantiates a `ThreadedDecoder`. Their NVDEC contexts race — when one actor's context is destroyed, another actor's background thread is still mid-operation on the same GPU and crashes.

There is also a hardware-level NVDEC instance limit (typically 2–5 per GPU). Spawning more `ThreadedDecoder` instances than available NVDEC engines causes allocation failures.

**Fix (applied 2026-04-25):** Switch Ray actor decode to `SimpleDecoder` instead of `ThreadedDecoder`:
- `SimpleDecoder` is synchronous — no background thread, no persistent CUDA context held between calls
- Safe to run concurrently on a shared GPU
- Better fit for our use case (random frame sampling, not full sequential decode)
- Throughput impact negligible: CLIP inference dominates per-video time (~50-100ms), decode is ~5ms either way

**Implementation:** Split `decode.py` into two functions:
- `decode_video()` — uses `ThreadedDecoder`, for standalone/profiling use
- `decode_video_actor()` — uses `SimpleDecoder`, called by all Ray workers

`embed.py` and `score.py` call `decode_video_actor()` and keep `ACTORS_PER_GPU=4` (16 concurrent actors across 4 GPUs).
`caption.py` uses `num_gpus=1` (full GPU per actor) so was never affected.

**Other solutions considered:**
- Shared explicit CUDA context (pycuda) — correct but adds pycuda dependency
- `num_gpus=1` per actor — works but wastes GPU packing (4→16 concurrent tasks)
- `force_cpu=True` PyAV decode — safe but unnecessary performance loss

**Status:** Fix applied 2026-04-25.