"""
decode.py
---------
Two GPU decode backends + CPU fallback:

  decode_video()       — ThreadedDecoder (standalone / profiling use)
  decode_video_actor() — SimpleDecoder   (Ray actors, fractional GPU sharing)

Why two functions:
  ThreadedDecoder keeps a background decode thread that holds a CUDA context
  for its lifetime. When multiple actors share a GPU (num_gpus=0.25, 4 actors
  per GPU), concurrent ThreadedDecoder instances race on NVDEC contexts and
  crash with CUDA_ERROR_CONTEXT_IS_DESTROYED.

  SimpleDecoder has no background thread — it decodes synchronously on demand.
  Safe to run concurrently on a shared GPU. For our use case (sampling 8 frames
  from a 10s clip) it's also a better fit: we do random seeks, not full sequential
  decode. Throughput impact is negligible since CLIP inference dominates.

Falls back to PyAV (ffmpeg) then OpenCV if PyNvVideoCodec is unavailable.

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

    # Convert before del dec — frames hold live GPU pointers into decoder's context.
    tensors = [torch.from_dlpack(f).to(device) for f in all_f]
    del dec, all_f

    batches = []
    for i in range(0, len(tensors), batch_size):
        batches.append(torch.stack(tensors[i : i + batch_size]))

    torch.cuda.synchronize()
    return batches, time.perf_counter() - t0


def decode_gpu_simple(video_path: str, max_frames: int = 512,
                      batch_size: int = 16, device: str = "cuda:0") -> tuple[list, float]:
    """
    SimpleDecoder — synchronous, no background thread, safe for concurrent use
    on a shared GPU (Ray fractional actors, num_gpus=0.25).
    Returns (batches, decode_time_s) — batches: list of [B,H,W,C] uint8 GPU tensors.
    """
    import logging
    log = logging.getLogger("decode")

    from PyNvVideoCodec import SimpleDecoder

    t0 = time.perf_counter()
    torch.cuda.synchronize()

    dec    = SimpleDecoder(video_path, gpu_id=0)
    frames = []
    for frame in dec:
        frames.append(frame)
        if len(frames) >= max_frames:
            break

    log.debug(f"SimpleDecoder: {len(frames)} frames  type={type(frames[0]) if frames else 'none'}")

    if frames:
        f0 = frames[0]
        log.debug(f"  frame[0] type={type(f0)}  attrs={[a for a in dir(f0) if not a.startswith('_')]}")

    # Convert to tensors while decoder (and its CUDA context) is still alive.
    # DecodedFrame objects hold live pointers into the decoder's GPU memory —
    # del dec before conversion causes CUDA_ERROR_CONTEXT_IS_DESTROYED.
    tensors = [torch.from_dlpack(f).to(device) for f in frames]
    del dec, frames

    batches = []
    for i in range(0, len(tensors), batch_size):
        batches.append(torch.stack(tensors[i : i + batch_size]))

    torch.cuda.synchronize()
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
    Ray actor decode — uses SimpleDecoder (no background thread, safe for
    concurrent fractional GPU actors). Returns (batches, decode_time_s, backend_used).
    """
    try:
        import PyNvVideoCodec  # noqa: F401
        batches, t = decode_gpu_simple(video_path, max_frames, batch_size, device)
        return batches, t, "pynvvideocodec_simple"
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