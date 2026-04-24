"""
embed.py
--------
Extract CLIP (ViT-B/32) embeddings for a set of videos.
Samples N frames per video, runs them through CLIP, stores per-video
mean embedding + per-frame embeddings in a .npz file.

Design:
  - Uses Ray to parallelize across videos (one Ray task per video)
  - Each task requests 0.25 GPU → 4 tasks run concurrently per GPU
  - With 4 GPUs: 16 concurrent embed tasks
  - Results written to data/embeddings/<video_stem>.npz

Usage:
  python embed.py --video_dir data/raw_videos --out_dir data/embeddings --frames_per_video 8 --num_gpus 4
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import ray
import wandb
from transformers import CLIPProcessor, CLIPModel

from decode import decode_video


MODEL_ID = "openai/clip-vit-base-patch32"


def _load_clip(device: str):
    model     = CLIPModel.from_pretrained(MODEL_ID).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model.eval()
    return model, processor


def _sample_frames(batches: list, n: int, is_gpu: bool) -> list:
    """Uniformly sample n frames from decoded batches."""
    all_frames = []
    for b in batches:
        if is_gpu:
            for i in range(b.shape[0]):
                all_frames.append(b[i].cpu().numpy())   # [H, W, C] uint8
        else:
            all_frames.extend(b)                         # list of [H, W, C] uint8 numpy

    if len(all_frames) == 0:
        return []
    indices = np.linspace(0, len(all_frames) - 1, min(n, len(all_frames)), dtype=int)
    return [all_frames[i] for i in indices]


@ray.remote(num_gpus=0.25)
def embed_video(video_path: str, out_dir: str,
                frames_per_video: int = 8) -> dict:
    """
    Ray task: decode + embed one video.
    Requests 0.25 GPU → 4 tasks run concurrently per GPU.
    With 4 GPUs: 16 concurrent tasks.
    Ray assigns the GPU via CUDA_VISIBLE_DEVICES — always use cuda:0 inside the task.
    """
    out_path = Path(out_dir) / (Path(video_path).stem + ".npz")
    if out_path.exists():
        return {"path": video_path, "status": "cached", "n_frames": 0, "time_s": 0.0}

    device = "cuda:0"  # Ray sets CUDA_VISIBLE_DEVICES per task
    t0 = time.perf_counter()
    try:
        batches, _, backend = decode_video(video_path, max_frames=64,
                                           batch_size=16, device=device)
        is_gpu   = backend != "opencv_cpu"
        frames   = _sample_frames(batches, frames_per_video, is_gpu)
        if not frames:
            return {"path": video_path, "status": "failed: no frames", "n_frames": 0, "time_s": 0.0}

        model, processor = _load_clip(device)
        inputs = processor(images=frames, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            feats = model.get_image_features(**inputs)          # [N, 512]
            feats = F.normalize(feats, dim=-1)

        feats_np = feats.cpu().numpy()
        mean_emb = feats_np.mean(axis=0)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        np.savez(out_path,
                 mean_embedding=mean_emb,       # [512]
                 frame_embeddings=feats_np,     # [N, 512]
                 video_path=np.array([video_path]))

        elapsed = time.perf_counter() - t0
        return {"path": video_path, "status": "ok",
                "n_frames": len(frames), "time_s": elapsed}

    except Exception as e:
        return {"path": video_path, "status": f"failed: {e}", "n_frames": 0, "time_s": 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir",         required=True)
    ap.add_argument("--out_dir",           default="data/embeddings")
    ap.add_argument("--frames_per_video",  type=int, default=8)
    ap.add_argument("--num_gpus",          type=int, default=1,
                    help="Number of GPUs available. Ray will spread tasks across them.")
    ap.add_argument("--ray_address",       default=None)
    args = ap.parse_args()

    video_dir = Path(args.video_dir)
    videos    = sorted(video_dir.rglob("*.mp4"))
    print(f"Found {len(videos)} videos in {video_dir}")
    # 0.25 GPU per task → 4 tasks/GPU × num_gpus concurrent tasks total
    print(f"Concurrency: {args.num_gpus * 4} tasks across {args.num_gpus} GPU(s)")

    wandb.init(project="video-curation", entity="rlx-labs",
               name="embed", resume="allow", id="embed-stage")

    ray.init(num_gpus=args.num_gpus, ignore_reinit_error=True)

    futures = [
        embed_video.remote(str(v), args.out_dir, args.frames_per_video)
        for v in videos
    ]

    ok = failed = cached = 0
    total = len(videos)
    pending = list(futures)
    t0 = time.perf_counter()

    while pending:
        done, pending = ray.wait(pending, num_returns=min(50, len(pending)), timeout=60)
        for fut in ray.get(done):
            s = fut["status"]
            if   s == "ok":     ok     += 1
            elif s == "cached": cached += 1
            else:               failed += 1
        completed = ok + failed + cached
        elapsed = time.perf_counter() - t0
        print(f"  [{completed}/{total}]  ok={ok}  cached={cached}  "
              f"failed={failed}  elapsed={elapsed:.1f}s")
        wandb.log({"embed/ok": ok, "embed/failed": failed, "embed/cached": cached,
                   "embed/completed": completed, "embed/total": total,
                   "embed/elapsed_s": elapsed})

    print(f"\nEmbeddings saved → {args.out_dir}")
    wandb.finish()
    ray.shutdown()


if __name__ == "__main__":
    main()
