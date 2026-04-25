"""
decode.py
---------
GPU-accelerated video decode using PyNvVideoCodec (ThreadedDecoder).
Reuses the same decode strategy from benchmark.py:
  - Decode all frames once into GPU tensors
  - Return as list of [B, H, W, C] uint8 CUDA tensors

Falls back to OpenCV CPU decode if PyNvVideoCodec is unavailable.

Usage (standalone):
  python decode.py --video data/raw_videos/playing_guitar/abc123_000010.mp4
"""

import argparse
import time
from pathlib import Path

import torch


def decode_gpu(video_path: str, max_frames: int = 512,
               batch_size: int = 16, device: str = "cuda:0") -> tuple[list, float]:
    """
    Decode up to max_frames from video_path using PyNvVideoCodec ThreadedDecoder.
    Returns (batches, decode_time_s) where batches is a list of [B,H,W,C] uint8 GPU tensors.

    The decoder is explicitly deleted before returning so its background C++ threads
    release the CUDA context immediately. This prevents CUDA_ERROR_CONTEXT_IS_DESTROYED
    when multiple Ray actors share a GPU (fractional GPU allocation).
    """
    from PyNvVideoCodec import ThreadedDecoder, OutputColorType

    t0 = time.perf_counter()
    torch.cuda.synchronize()

    dec   = ThreadedDecoder(video_path, max_frames * 2, gpu_id=0,
                            output_color_type=OutputColorType.RGB)
    all_f = dec.get_batch_frames(max_frames)
    # Copy frames to CPU numpy before deleting the decoder, since the decoder
    # owns the GPU memory backing those frame objects.
    frames_np = [f.numpy() if hasattr(f, "numpy") else f for f in all_f]
    del dec  # force C++ destructor now — releases CUDA context before any other actor uses it

    batches = []
    for i in range(0, len(frames_np), batch_size):
        chunk = frames_np[i : i + batch_size]
        batch = torch.stack([
            torch.as_tensor(f, device=device).clone() for f in chunk
        ])  # [B, H, W, C] uint8
        batches.append(batch)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return batches, elapsed


def decode_cpu(video_path: str, max_frames: int = 512,
               batch_size: int = 16) -> tuple[list, float]:
    """
    Fallback CPU decode using OpenCV.
    Returns (batches, decode_time_s) where batches is a list of numpy arrays [B,H,W,C].
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

    batches = []
    for i in range(0, len(frames), batch_size):
        batches.append(frames[i : i + batch_size])

    elapsed = time.perf_counter() - t0
    return batches, elapsed


def decode_video(video_path: str, max_frames: int = 512,
                 batch_size: int = 16, device: str = "cuda:0") -> tuple[list, float, str]:
    """
    Decode video, preferring GPU. Returns (batches, decode_time_s, backend_used).
    Falls back to CPU only if PyNvVideoCodec is not importable (not installed).
    """
    try:
        import PyNvVideoCodec  # noqa: F401 — probe import only
    except ImportError:
        batches, t = decode_cpu(video_path, max_frames, batch_size)
        return batches, t, "opencv_cpu"

    batches, t = decode_gpu(video_path, max_frames, batch_size, device)
    return batches, t, "pynvvideocodec"


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
