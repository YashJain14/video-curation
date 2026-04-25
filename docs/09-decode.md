# Decode Library — decode.py

Provides two decode paths: one for Ray actors (CPU/PyAV) and one for standalone use (GPU/PyNvVideoCodec). Both return the same interface: `(batches, decode_time_s, backend_used)`.

---

## Two Decode Paths

| Function              | Backend              | When to use                              |
|-----------------------|----------------------|------------------------------------------|
| `decode_video()`      | PyNvVideoCodec (GPU) | Standalone profiler (`profile_run.py`)   |
| `decode_video_actor()`| PyAV (CPU)           | Ray actors in embed.py and score.py      |

---

## `decode_video_actor()` — CPU path for Ray actors

```python
def decode_video_actor(video_path, max_frames=512, batch_size=16, device="cuda:0"):
    # Always uses PyAV. Falls back to OpenCV if av not installed.
    batches, t = decode_pyav(video_path, max_frames, batch_size)
    return batches, t, "pyav"
```

**Returns:** `(batches, decode_time_s, backend_used)`  
`batches` = list of `list[np.ndarray]`, each array `[H, W, C]` uint8 RGB.

### Why CPU for Ray actors

With `num_gpus=0.25`, four actors share one physical GPU. Every PyNvVideoCodec decoder instance creates its own CUDA context on the shared GPU via the NVDEC hardware engine. Concurrent contexts race:

- One actor destroying its context while another holds live DLPack pointers into shared GPU memory.
- Surfaces as `CUDA_ERROR_CONTEXT_IS_DESTROYED` in `ExternalBuffer.cpp`.

PyAV decodes entirely on CPU and returns self-contained `numpy` arrays — no shared GPU state to race on.

**Throughput cost is negligible:** CLIP inference takes ~50–100ms per video; CPU decode of 8 frames takes ~3ms. Decode is never the bottleneck.

---

## `decode_video()` — GPU path for standalone use

```python
def decode_video(video_path, max_frames=512, batch_size=16, device="cuda:0"):
    # Tries: PyNvVideoCodec ThreadedDecoder → PyAV → OpenCV
    batches, t = decode_gpu_threaded(video_path, max_frames, batch_size, device)
    return batches, t, "pynvvideocodec_threaded"
```

**Returns:** `(batches, decode_time_s, backend_used)`  
`batches` = list of `torch.Tensor` `[B, H, W, C]` uint8 on GPU.

Used only by `profile_run.py` where one process owns the full GPU.

---

## Backend Details

### `decode_pyav()` — CPU ffmpeg

```python
import av
with av.open(video_path) as container:
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
```

Returns `list[np.ndarray [H,W,C]]` grouped into batches. Pure CPU, thread-safe.

### `decode_gpu_threaded()` — PyNvVideoCodec ThreadedDecoder

Uses a background decode thread for maximum throughput. Unsafe for concurrent use.

**DLPack safety pattern required:**
```python
tensors = [torch.from_dlpack(f).clone() for f in all_f]
torch.cuda.synchronize()   # BEFORE del dec — clone is async
del dec, all_f
```

`from_dlpack` is zero-copy — the decoder owns the source memory. `.clone()` forces an owned copy. `synchronize()` must happen before `del dec`, not after — otherwise the async clone may still be in flight when the decoder's GPU memory is freed.

### `decode_gpu_simple()` — PyNvVideoCodec SimpleDecoder

Synchronous iterator-based decoder. Same DLPack safety pattern as ThreadedDecoder.

### `decode_cpu()` — OpenCV fallback

Last-resort fallback when neither PyAV nor PyNvVideoCodec is installed.

---

## Bug History — Why We Use CPU Decode for Actors

The full sequence of bugs encountered while trying to use GPU decode in fractional-GPU Ray actors:

| Bug | Symptom | Root cause | Fix |
|-----|---------|-----------|-----|
| Bug 1 | `BaseModelOutputWithPooling has no attribute 'norm'` — 100% silent CLIP failures | `CLIPModel.get_image_features()` returns wrong type on installed transformers version | Call sub-modules directly: `vision_model()` → `visual_projection()` |
| Bug 2 | SIGSEGV in `ThreadedDecoder.end()` after ~750 videos | 4 actors per GPU race on NVDEC CUDA contexts | Switch to `SimpleDecoder` (synchronous) |
| Bug 3 | `can't convert np.ndarray of type numpy.object_` | `SimpleDecoder` returns DLPack objects, not numpy arrays | Convert via `torch.from_dlpack(f)` instead of `np.array(f)` |
| Bug 4 | `CUDA_ERROR_CONTEXT_IS_DESTROYED` in `ExternalBuffer.cpp` | `del dec` freed GPU memory while async clone still in flight | Call `.clone()` before `del dec`; synchronize before delete |
| Bug 5 | Same crash even with `.clone()` | DLPack zero-copy clone races PyNvVideoCodec's internal stream; `synchronize()` was after `del dec`, too late | **Switch Ray actors to PyAV (CPU decode) permanently** |

---

## Future Path

If CPU decode ever becomes a bottleneck, the correct fix is not to re-introduce PyNvVideoCodec in actors. Instead:

1. **Pass PyTorch's CUDA stream explicitly** to `SimpleDecoder`:
   ```python
   SimpleDecoder(path, gpu_id=0,
                 cuda_context=int(cuCtxGetCurrent()),
                 cuda_stream=stream.cuda_stream)
   ```
2. **Migrate to `torchcodec`** (Meta's NVDEC wrapper that handles DLPack synchronization correctly).

---

## CLI (standalone decode test)

```bash
python decode.py \
  --video      $SCRATCH_DIR/raw_videos/playing_guitar/abc123_000010.mp4 \
  --max_frames 64 \
  --batch_size 16 \
  --device     cuda:0
```

Prints backend, frame count, batch count, and decode time.
