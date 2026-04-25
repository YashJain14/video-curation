---
name: video-curation pipeline bugs
description: Three bugs found in the embed stage during April 2026 debugging session; Bug 1 confirmed fixed, Bug 2 fix applied but unverified, Bug 3 fix applied but unverified
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

**Status:** STILL OCCURRING after switching to SimpleDecoder. See Bug 3 and Bug 4 for the follow-on issues. SimpleDecoder alone does not solve the CUDA context destruction problem — the frames hold live GPU pointers into the decoder's context, so the context must stay alive until after tensor conversion.

---

## Bug 3 — `SimpleDecoder` frames can't be converted via `np.array(f)` (dtype=object)

**Symptom:** 100% of videos fail with:
```
failed: can't convert np.ndarray of type numpy.object_. The only supported types are: float64, float32, ...
```
This happened immediately after switching to `SimpleDecoder` (Bug 2 fix).

**Root cause:** `SimpleDecoder` (and `ThreadedDecoder`) return DLPack-compatible frame objects, not raw numpy arrays. Calling `np.array(f)` wraps the object itself rather than extracting pixel data, producing a 0-d array of dtype=object. `torch.as_tensor()` then rejects it.

**Fix (applied 2026-04-25):** Use `torch.from_dlpack(f).to(device)` instead of `torch.as_tensor(np.array(f), device=device).clone()` in both `decode_gpu_simple` and `decode_gpu_threaded` in `decode.py`.

**How to apply:** Any PyNvVideoCodec frame object (from either decoder) must be converted via DLPack: `torch.from_dlpack(frame)`. Never pass directly to `np.array()`.

**Status:** Fix applied 2026-04-25. UNVERIFIED — see Bug 4 below which was diagnosed in the same run.

---

## Bug 4 — `DecodedFrame` holds live GPU pointer; `del dec` before conversion causes `CUDA_ERROR_CONTEXT_IS_DESTROYED`

**Symptom:** `CUDA_ERROR_CONTEXT_IS_DESTROYED` or `CUDA_ERROR_INVALID_CONTEXT` in `ExternalBuffer.cpp:131` when converting frames to tensors. Crashes actors even with `SimpleDecoder`.

**Root cause:** `DecodedFrame` objects returned by `SimpleDecoder` (and `ThreadedDecoder`) are not self-contained copies — they hold live pointers into GPU memory that is owned by the decoder's CUDA context. Calling `del dec` before converting frames to tensors destroys the context while the frame pointers are still live. The crash fires at `torch.from_dlpack(f)` or similar when those pointers are dereferenced.

Confirmed by the log: the crash in `ExternalBuffer.cpp` happened immediately after `del dec` and before tensor conversion completed.

**Root cause (deeper):** `torch.from_dlpack()` is zero-copy — it wraps the decoder's GPU pointer without copying data. Even converting before `del dec` is not enough, because the resulting tensors still point into decoder-owned memory. When `del dec` frees that memory (or another actor reuses the GPU buffer), any access to those tensors causes `illegal memory access`.

**Fix (applied 2026-04-25):** Convert and immediately `.clone()` to force a copy into PyTorch-owned GPU memory before deleting the decoder:
```python
tensors = [torch.from_dlpack(f).to(device).clone() for f in frames]
del dec, frames   # safe: tensors now own their GPU memory
```

**How to apply:** After `torch.from_dlpack()` on any PyNvVideoCodec frame, always call `.clone()` before the decoder goes out of scope. The dlpack tensor is just a pointer alias, not a copy.

**Status:** Fix applied 2026-04-25. UNVERIFIED — awaiting next job run on cluster.