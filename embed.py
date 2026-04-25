"""
embed.py
--------
Extract CLIP (ViT-B/32) embeddings for a set of videos.
Samples N frames per video, runs them through CLIP, stores per-video
mean embedding + per-frame embeddings in a .npz file.

Design:
  - One Ray actor per GPU slot (4 actors × 4 GPUs = 16 actors)
  - Each actor loads CLIP once, then processes all assigned videos
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
ACTORS_PER_GPU = 4


def _sample_frames(batches: list, n: int, is_gpu: bool) -> list:
    all_frames = []
    for b in batches:
        if is_gpu:
            for i in range(b.shape[0]):
                all_frames.append(b[i].cpu().numpy())
        else:
            all_frames.extend(b)
    if not all_frames:
        return []
    indices = np.linspace(0, len(all_frames) - 1, min(n, len(all_frames)), dtype=int)
    return [all_frames[i] for i in indices]


@ray.remote(num_gpus=1.0 / ACTORS_PER_GPU)
class EmbedWorker:
    def __init__(self):
        self.device = "cuda:0"
        model = CLIPModel.from_pretrained(MODEL_ID).to(self.device)
        model.eval()
        self.model = model
        self.processor = CLIPProcessor.from_pretrained(MODEL_ID)

    def process(self, video_path: str, out_dir: str, frames_per_video: int) -> dict:
        out_path = Path(out_dir) / (Path(video_path).stem + ".npz")
        if out_path.exists():
            return {"path": video_path, "status": "cached", "n_frames": 0, "time_s": 0.0}

        t0 = time.perf_counter()
        try:
            batches, _, backend = decode_video(video_path, max_frames=64,
                                               batch_size=16, device=self.device)
            is_gpu = backend != "opencv_cpu"
            frames = _sample_frames(batches, frames_per_video, is_gpu)
            if not frames:
                return {"path": video_path, "status": "failed: no frames",
                        "n_frames": 0, "time_s": 0.0}

            inputs = self.processor(images=frames, return_tensors="pt", padding=True).to(self.device)
            with torch.inference_mode():
                vision_out = self.model.vision_model(pixel_values=inputs["pixel_values"])
                feats = self.model.visual_projection(vision_out.pooler_output)
                feats = F.normalize(feats, dim=-1)

            feats_np = feats.cpu().numpy()
            mean_emb = feats_np.mean(axis=0)
            mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)

            Path(out_dir).mkdir(parents=True, exist_ok=True)
            np.savez(out_path,
                     mean_embedding=mean_emb,
                     frame_embeddings=feats_np,
                     video_path=np.array([video_path], dtype=str))

            return {"path": video_path, "status": "ok",
                    "n_frames": len(frames), "time_s": time.perf_counter() - t0}

        except Exception as e:
            return {"path": video_path, "status": f"failed: {e}",
                    "n_frames": 0, "time_s": 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir",        required=True)
    ap.add_argument("--out_dir",          default="data/embeddings")
    ap.add_argument("--frames_per_video", type=int, default=8)
    ap.add_argument("--num_gpus",         type=int, default=1)
    ap.add_argument("--ray_address",      default=None)
    args = ap.parse_args()

    video_dir = Path(args.video_dir)
    videos    = sorted(video_dir.rglob("*.mp4"))
    n_actors  = args.num_gpus * ACTORS_PER_GPU
    print(f"Found {len(videos)} videos in {video_dir}")
    print(f"Actors: {n_actors} ({ACTORS_PER_GPU} per GPU × {args.num_gpus} GPUs)")

    wandb.init(project="video-curation", entity="rlx-labs",
               name="embed-stage",
               settings=wandb.Settings(init_timeout=300))

    ray.init(num_gpus=args.num_gpus, ignore_reinit_error=True)

    workers = [EmbedWorker.remote() for _ in range(n_actors)]

    # Round-robin dispatch
    futures = [
        workers[i % n_actors].process.remote(str(v), args.out_dir, args.frames_per_video)
        for i, v in enumerate(videos)
    ]

    ok = failed = cached = 0
    total   = len(videos)
    pending = list(futures)
    t0      = time.perf_counter()

    first_errors: list[str] = []
    while pending:
        done, pending = ray.wait(pending, num_returns=min(50, len(pending)), timeout=60)
        for fut in ray.get(done):
            s = fut["status"]
            if   s == "ok":     ok     += 1
            elif s == "cached": cached += 1
            else:
                failed += 1
                if len(first_errors) < 5:
                    first_errors.append(f"{Path(fut['path']).name}: {s}")
        completed = ok + failed + cached
        elapsed   = time.perf_counter() - t0
        print(f"  [{completed}/{total}]  ok={ok}  cached={cached}  "
              f"failed={failed}  elapsed={elapsed:.1f}s")
        if first_errors:
            for msg in first_errors:
                print(f"    ERROR: {msg}")
            first_errors.clear()
        wandb.log({"embed/ok": ok, "embed/failed": failed, "embed/cached": cached,
                   "embed/completed": completed, "embed/total": total,
                   "embed/elapsed_s": elapsed})

    print(f"\nEmbeddings saved → {args.out_dir}")
    wandb.finish()
    ray.shutdown()


if __name__ == "__main__":
    main()
