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
  python score.py --video_dir data/raw_videos --emb_dir data/embeddings \
                  --out data/scores.json --min_score 4.5
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

from decode import decode_video


CLIP_MODEL_ID = "openai/clip-vit-large-patch14"


class AestheticMLP(nn.Module):
    """
    LAION improved aesthetic predictor MLP.
    Input: 768-dim CLIP ViT-L/14 embedding
    Output: scalar score
    """
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


def load_aesthetic_model(device: str) -> AestheticMLP:
    """
    Load LAION aesthetic predictor weights.
    Downloads from HuggingFace on first run, cached after.
    """
    from huggingface_hub import hf_hub_download
    weights_path = hf_hub_download(
        repo_id="shunk031/improved-aesthetic-predictor",
        filename="sac+logos+ava1-l14-linearMSE.pth",
    )
    state = torch.load(weights_path, map_location=device)
    model = AestheticMLP().to(device)
    model.load_state_dict(state)
    model.eval()
    return model


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


@ray.remote(num_gpus=0.25)
def score_video(video_path: str, frames_per_video: int = 8) -> dict:
    """
    Ray task: decode + score one video aesthetically.
    0.25 GPU per task → 4 tasks/GPU × 4 GPUs = 16 concurrent tasks.
    Ray sets CUDA_VISIBLE_DEVICES per task — always use cuda:0 inside.
    """
    device = "cuda:0"
    t0 = time.perf_counter()
    try:
        batches, _, backend = decode_video(video_path, max_frames=64,
                                           batch_size=16, device=device)
        is_gpu = backend != "opencv_cpu"
        frames = _sample_frames(batches, frames_per_video, is_gpu)
        if not frames:
            return {"path": video_path, "status": "failed: no frames",
                    "mean_score": 0.0, "frame_scores": []}

        clip_model     = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
        clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        clip_model.eval()
        aesthetic_model = load_aesthetic_model(device)

        inputs = clip_processor(images=frames, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feats  = clip_model.get_image_features(**inputs)        # [N, 768]
            feats  = F.normalize(feats, dim=-1)
            scores = aesthetic_model(feats).squeeze(-1).cpu().numpy()  # [N]

        mean_score   = float(scores.mean())
        frame_scores = scores.tolist()

        return {
            "path":         video_path,
            "status":       "ok",
            "mean_score":   mean_score,
            "frame_scores": frame_scores,
            "time_s":       time.perf_counter() - t0,
        }
    except Exception as e:
        return {"path": video_path, "status": f"failed: {e}",
                "mean_score": 0.0, "frame_scores": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir",        required=True)
    ap.add_argument("--out",              default="data/scores.json")
    ap.add_argument("--frames_per_video", type=int,   default=8)
    ap.add_argument("--min_score",        type=float, default=4.5,
                    help="Print how many clips pass this threshold")
    ap.add_argument("--num_gpus",         type=int,   default=1)
    ap.add_argument("--ray_address",      default=None)
    args = ap.parse_args()

    videos = sorted(Path(args.video_dir).rglob("*.mp4"))
    print(f"Scoring {len(videos)} videos ...")
    print(f"Concurrency: {args.num_gpus * 4} tasks across {args.num_gpus} GPU(s)")

    wandb.init(project="video-curation", entity="rlx-labs",
               name="score", resume="allow", id="score-stage")

    ray.init(num_gpus=args.num_gpus, ignore_reinit_error=True)

    futures = [
        score_video.remote(str(v), args.frames_per_video)
        for v in videos
    ]

    results = []
    ok = failed = 0
    total = len(videos)
    pending = list(futures)
    t0 = time.perf_counter()

    while pending:
        done, pending = ray.wait(pending, num_returns=min(50, len(pending)), timeout=60)
        for res in ray.get(done):
            results.append(res)
            if res["status"] == "ok": ok     += 1
            else:                     failed += 1
        completed = ok + failed
        elapsed = time.perf_counter() - t0
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
