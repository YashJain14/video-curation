"""
decode.py
---------
Two decode paths:

  decode_video()       — GPU (ThreadedDecoder/PyNvVideoCodec) for standalone use
  decode_video_actor() — CPU (PyAV) for Ray actors with fractional GPU sharing

Why the actor path is CPU:
  Every PyNvVideoCodec decoder (Threaded or Simple) creates its own CUDA
  context on the GPU. With num_gpus=0.25 (4 actors per GPU), concurrent
  decoders race on NVDEC, and DLPack frame pointers become invalid when any
  actor's context is destroyed → CUDA_ERROR_CONTEXT_IS_DESTROYED. PyAV runs
  on CPU and returns self-contained numpy frames, so there is no shared GPU
  state to race on.

  The throughput cost is negligible for this pipeline: CLIP inference
  dominates per-video time (~50–100 ms), CPU decode of 8 sampled frames is
  a few ms. We keep the GPU decoder path for the standalone profiler, where
  one process owns the GPU.

Usage (standalone):
  python decode.py --video data/raw_videos/playing_guitar/abc123_000010.mp4
"""

import argparse
import time

import numpy as np
import torch


def decode_gpu_threaded(video_path: str, max_frames: int = 512,
                        batch_size: int = 16, device: str = "cuda:0") -> tuple[list, float]:
    """
    ThreadedDecoder — fast sequential decode, standalone use only.
    NOT safe for concurrent use on a shared GPU (Ray fractional actors).
    Returns (batches, decode_time_s) — batches: list of [B,H,W,C] uint8 GPU tensors.
    """
    from PyNvVideoCodec import ThreadedDecoder, OutputColorType

    t0 = time.perf_counter()
    torch.cuda.synchronize()

    dec   = ThreadedDecoder(video_path, max_frames * 2, gpu_id=0,
                            output_color_type=OutputColorType.RGB)
    all_f = dec.get_batch_frames(max_frames)

    # from_dlpack is zero-copy; clone forces an owned copy. Synchronize BEFORE
    # del dec — the clone is async and the decoder owns the source memory.
    tensors = [torch.from_dlpack(f).clone() for f in all_f]
    torch.cuda.synchronize()
    del dec, all_f

    batches = []
    for i in range(0, len(tensors), batch_size):
        batches.append(torch.stack(tensors[i : i + batch_size]))

    return batches, time.perf_counter() - t0


def decode_gpu_simple(video_path: str, max_frames: int = 512,
                      batch_size: int = 16, device: str = "cuda:0") -> tuple[list, float]:
    """
    SimpleDecoder — synchronous, no background thread.
    Standalone use only — see decode_video_actor() for the Ray-actor path,
    which uses PyAV CPU decode to avoid CUDA-context contention on shared GPUs.
    Returns (batches, decode_time_s) — batches: list of [B,H,W,C] uint8 GPU tensors.
    """
    from PyNvVideoCodec import SimpleDecoder

    t0 = time.perf_counter()
    torch.cuda.synchronize()

    dec    = SimpleDecoder(video_path, gpu_id=0)
    frames = []
    for frame in dec:
        frames.append(frame)
        if len(frames) >= max_frames:
            break

    # from_dlpack is zero-copy; clone forces an owned copy. Synchronize BEFORE
    # del dec — clone is async and the decoder owns the source GPU memory until
    # the copy actually executes.
    tensors = [torch.from_dlpack(f).clone() for f in frames]
    torch.cuda.synchronize()
    del dec, frames

    batches = []
    for i in range(0, len(tensors), batch_size):
        batches.append(torch.stack(tensors[i : i + batch_size]))

    return batches, time.perf_counter() - t0


def decode_pyav(video_path: str, max_frames: int = 512,
                batch_size: int = 16) -> tuple[list, float]:
    """
    CPU decode via PyAV (ffmpeg). Fallback if PyNvVideoCodec is unavailable.
    Returns (batches, decode_time_s) — batches: list of numpy arrays [B,H,W,C].
    """
    import av

    t0 = time.perf_counter()
    frames = []
    with av.open(video_path) as container:
        for frame in container.decode(video=0):
            if len(frames) >= max_frames:
                break
            frames.append(frame.to_ndarray(format="rgb24"))

    batches = [frames[i : i + batch_size] for i in range(0, len(frames), batch_size)]
    return batches, time.perf_counter() - t0


def decode_cpu(video_path: str, max_frames: int = 512,
               batch_size: int = 16) -> tuple[list, float]:
    """
    CPU decode via OpenCV. Last-resort fallback.
    Returns (batches, decode_time_s) — batches: list of numpy arrays [B,H,W,C].
    """
    import cv2

    t0 = time.perf_counter()
    cap    = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()

    batches = [frames[i : i + batch_size] for i in range(0, len(frames), batch_size)]
    return batches, time.perf_counter() - t0


def decode_video(video_path: str, max_frames: int = 512,
                 batch_size: int = 16, device: str = "cuda:0") -> tuple[list, float, str]:
    """
    Standalone decode — uses ThreadedDecoder for maximum throughput.
    Returns (batches, decode_time_s, backend_used).
    """
    try:
        import PyNvVideoCodec  # noqa: F401
        batches, t = decode_gpu_threaded(video_path, max_frames, batch_size, device)
        return batches, t, "pynvvideocodec_threaded"
    except ImportError:
        pass

    try:
        import av  # noqa: F401
        batches, t = decode_pyav(video_path, max_frames, batch_size)
        return batches, t, "pyav"
    except ImportError:
        pass

    batches, t = decode_cpu(video_path, max_frames, batch_size)
    return batches, t, "opencv_cpu"


def decode_video_actor(video_path: str, max_frames: int = 512,
                       batch_size: int = 16, device: str = "cuda:0") -> tuple[list, float, str]:
    """
    Ray actor decode — PyAV (CPU) by default.

    Why CPU decode for fractional-GPU actors:
      With num_gpus=0.25 (4 actors per GPU), every PyNvVideoCodec decoder
      instance creates its own CUDA context on the shared GPU. Concurrent
      contexts race on NVDEC, and DLPack frame pointers become invalid when
      any actor's context is destroyed — surfacing as CUDA_ERROR_CONTEXT_IS_DESTROYED
      mid-pipeline. PyAV decodes on CPU and hands back self-contained numpy
      frames, so there is no shared GPU state to race on.

      Throughput cost is negligible: CLIP inference dominates per-video time
      (~50–100 ms), CPU decode of 8 sampled frames is a few ms.

    Returns (batches, decode_time_s, backend_used).
    """
    try:
        import av  # noqa: F401
        batches, t = decode_pyav(video_path, max_frames, batch_size)
        return batches, t, "pyav"
    except ImportError:
        pass

    batches, t = decode_cpu(video_path, max_frames, batch_size)
    return batches, t, "opencv_cpu"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",      required=True)
    ap.add_argument("--max_frames", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device",     default="cuda:0")
    args = ap.parse_args()

    batches, elapsed, backend = decode_video(
        args.video, args.max_frames, args.batch_size, args.device
    )
    total_frames = sum(b.shape[0] if hasattr(b, "shape") else len(b) for b in batches)
    print(f"Backend : {backend}")
    print(f"Frames  : {total_frames}")
    print(f"Batches : {len(batches)}")
    print(f"Time    : {elapsed:.3f}s  ({total_frames/elapsed:.1f} fps)")