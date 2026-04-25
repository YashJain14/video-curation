"""
score.py
--------
Aesthetic quality scoring for video clips.
Uses the LAION aesthetic predictor (MLP head over CLIP ViT-L/14 embeddings).

For each video:
  - Sample N frames
  - Extract CLIP ViT-L/14 embeddings
  - Run LAION aesthetic MLP → score in [0, 10]
  - Store mean score + per-frame scores

Scores are used downstream to filter low-quality clips before shard writing.

LAION aesthetic predictor:
  - Trained on human aesthetic ratings from LAION-5B
  - Predicts a [0, 10] score; >5.0 is considered "good" for training data
  - Model weights: https://github.com/christophschuhmann/improved-aesthetic-predictor

Usage:
  python score.py --video_dir data/raw_videos --out data/scores.json --num_gpus 4
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ray
import wandb
from transformers import CLIPProcessor, CLIPModel

from decode import decode_video_actor


CLIP_MODEL_ID   = "openai/clip-vit-large-patch14"
ACTORS_PER_GPU  = 4
AESTHETIC_REPO  = "shunk031/improved-aesthetic-predictor"
AESTHETIC_FILE  = "sac+logos+ava1-l14-linearMSE.pth"


class AestheticMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


def _sample_frames(batches, n, is_gpu):
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
class ScoreWorker:
    def __init__(self, weights_path: str):
        self.device = "cuda:0"

        clip = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(self.device)
        clip.eval()
        self.clip      = clip
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

        state = torch.load(weights_path, map_location=self.device, weights_only=True)
        aes   = AestheticMLP().to(self.device)
        aes.load_state_dict(state)
        aes.eval()
        self.aesthetic = aes

    def process(self, video_path: str, frames_per_video: int, cache_dir: str) -> dict:
        cache_path = Path(cache_dir) / (Path(video_path).stem + ".score.json")
        if cache_path.exists():
            import json
            return json.loads(cache_path.read_text())

        t0 = time.perf_counter()
        try:
            batches, _, backend = decode_video_actor(video_path, max_frames=64,
                                                       batch_size=16, device=self.device)
            is_gpu = "pynvvideocodec" in backend
            frames = _sample_frames(batches, frames_per_video, is_gpu)
            if not frames:
                return {"path": video_path, "status": "failed: no frames",
                        "mean_score": 0.0, "frame_scores": []}

            inputs = self.processor(images=frames, return_tensors="pt", padding=True).to(self.device)
            with torch.inference_mode():
                vision_out = self.clip.vision_model(pixel_values=inputs["pixel_values"])
                feats      = self.clip.visual_projection(vision_out.pooler_output)
                feats      = F.normalize(feats, dim=-1)
                scores     = self.aesthetic(feats).squeeze(-1).cpu().numpy()

            result = {
                "path":         video_path,
                "status":       "ok",
                "mean_score":   float(scores.mean()),
                "frame_scores": scores.tolist(),
                "time_s":       time.perf_counter() - t0,
            }
            import json
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(result))
            return result
        except Exception as e:
            return {"path": video_path, "status": f"failed: {e}",
                    "mean_score": 0.0, "frame_scores": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir",        required=True)
    ap.add_argument("--out",              default="data/scores.json")
    ap.add_argument("--frames_per_video", type=int,   default=8)
    ap.add_argument("--min_score",        type=float, default=4.5)
    ap.add_argument("--num_gpus",         type=int,   default=1)
    ap.add_argument("--ray_address",      default=None)
    args = ap.parse_args()

    # Resolve weights path before Ray starts (avoids HF API calls inside workers)
    from huggingface_hub import hf_hub_download
    weights_path = hf_hub_download(repo_id=AESTHETIC_REPO, filename=AESTHETIC_FILE)
    print(f"Aesthetic weights: {weights_path}")

    videos   = sorted(Path(args.video_dir).rglob("*.mp4"))
    n_actors = args.num_gpus * ACTORS_PER_GPU
    print(f"Scoring {len(videos)} videos ...")
    print(f"Actors: {n_actors} ({ACTORS_PER_GPU} per GPU × {args.num_gpus} GPUs)")

    wandb.init(project="video-curation", entity="rlx-labs",
               name="score", resume="allow", id="score-stage")

    ray.init(num_gpus=args.num_gpus, ignore_reinit_error=True)

    workers = [ScoreWorker.remote(weights_path) for _ in range(n_actors)]

    cache_dir = str(Path(args.out).parent / "score_cache")
    futures = [
        workers[i % n_actors].process.remote(str(v), args.frames_per_video, cache_dir)
        for i, v in enumerate(videos)
    ]

    results = []
    ok = failed = 0
    total   = len(videos)
    pending = list(futures)
    t0      = time.perf_counter()

    while pending:
        done, pending = ray.wait(pending, num_returns=min(50, len(pending)), timeout=60)
        for res in ray.get(done):
            results.append(res)
            if res["status"] == "ok":
                ok += 1
            else:
                failed += 1
                print(f"    ERROR: {Path(res['path']).name}: {res['status']}")
        completed = ok + failed
        elapsed   = time.perf_counter() - t0
        print(f"  [{completed}/{total}]  ok={ok}  failed={failed}  elapsed={elapsed:.1f}s")
        wandb.log({"score/ok": ok, "score/failed": failed,
                   "score/completed": completed, "score/total": total,
                   "score/elapsed_s": elapsed})

    scores_vals = [r["mean_score"] for r in results if r["status"] == "ok"]
    passing     = sum(1 for s in scores_vals if s >= args.min_score)
    print(f"\nScore stats:")
    print(f"  Mean  : {np.mean(scores_vals):.2f}")
    print(f"  Median: {np.median(scores_vals):.2f}")
    print(f"  ≥{args.min_score}: {passing}/{len(scores_vals)}  ({100*passing/max(len(scores_vals),1):.1f}%)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nScores saved → {out}")
    wandb.finish()
    ray.shutdown()


if __name__ == "__main__":
    main()
