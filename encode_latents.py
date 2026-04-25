"""
encode_latents.py
-----------------
Pre-encode video frames into VAE latent tensors for training efficiency.

Why pre-encode latents:
  Video diffusion models (DiT-based architectures like CogVideoX, Open-Sora,
  Wan, HunyuanVideo) operate in latent space, not pixel space. During training,
  every batch must encode each video through the VAE before the diffusion forward
  pass. For a dataset of millions of clips, this encoding cost is paid repeatedly
  across all training epochs — wasting GPU cycles that could go to the diffusion
  model itself.

  Pre-encoding stores the latents once at data preparation time. At training,
  the DataLoader reads latent tensors directly from disk, skipping the VAE
  encoder entirely. This typically reduces training GPU time by 15–30% for
  video diffusion models, depending on spatial resolution and sequence length.

VAE used here:
  stabilityai/sd-vae-ft-mse — the Stable Diffusion VAE, commonly used as a
  spatial compressor for video DiTs (4x spatial downsampling, 4-channel latent).
  In a real pipeline, swap this for the VAE used by the target training codebase
  (e.g. CogVideoX uses a 3D causal VAE, HunyuanVideo uses its own VAE).

Encoding strategy:
  Each video is decoded into N evenly-spaced frames via PyAV (CPU), then batched
  through the VAE encoder on GPU. Latents are stored as float16 .pt tensors to
  minimise disk usage while preserving precision adequate for training.

  Output shape: [C, T, H//8, W//8] where
    C = 4  (VAE latent channels)
    T = num_frames  (temporal dimension, one latent per frame)
    H, W = frame spatial dims before 8× VAE downsampling

Output:
  latents/<video_stem>.pt   — torch.float16 tensor [4, T, H//8, W//8]
  Cache: skip if .pt already exists

Shard integration:
  shard.py reads latents/<stem>.pt and embeds it as <key>.latent.pt inside
  each WebDataset .tar shard, so the training DataLoader can read latents
  directly without a separate lookup step.

Usage:
  python encode_latents.py \\
    --video_dir  $SCRATCH_DIR/raw_videos \\
    --out_dir    $SCRATCH_DIR/latents \\
    --scores     $SCRATCH_DIR/scores.json \\
    --motion     $SCRATCH_DIR/motion_scores.json \\
    --captions_quality $SCRATCH_DIR/caption_quality.json \\
    --min_score  4.5 \\
    --min_motion 3.0 \\
    --min_quality 0.3 \\
    --num_gpus   4 \\
    --num_frames 16 \\
    --resolution 256
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import ray
import wandb


VAE_MODEL_ID   = "stabilityai/sd-vae-ft-mse"
ACTORS_PER_GPU = 2     # VAE encoder is lighter than CLIP; 2 actors per GPU is safe


def _decode_frames_pyav(video_path: str, num_frames: int,
                         resolution: int) -> torch.Tensor:
    """
    Decode num_frames evenly-spaced frames via PyAV and resize to resolution×resolution.
    Returns float32 tensor [T, C, H, W] normalised to [-1, 1].
    """
    import av
    import torchvision.transforms.functional as TF

    frames = []
    with av.open(video_path) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))

    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")

    indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
    selected = [frames[i] for i in indices]

    tensors = []
    for f in selected:
        t = torch.from_numpy(f).permute(2, 0, 1).float() / 255.0   # [3,H,W] in [0,1]
        t = TF.resize(t, [resolution, resolution],
                      interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        tensors.append(t * 2.0 - 1.0)   # normalise to [-1, 1]

    return torch.stack(tensors)   # [T, 3, H, W]


@ray.remote(num_gpus=1.0 / ACTORS_PER_GPU)
class LatentEncodeWorker:
    def __init__(self, resolution: int, num_frames: int):
        from diffusers import AutoencoderKL
        self.resolution  = resolution
        self.num_frames  = num_frames
        self.device      = "cuda:0"
        self.vae = AutoencoderKL.from_pretrained(VAE_MODEL_ID).to(self.device)
        self.vae.eval()

    def process(self, video_path: str, out_dir: str) -> dict:
        out_path = Path(out_dir) / (Path(video_path).stem + ".pt")
        if out_path.exists():
            return {"path": video_path, "status": "cached",
                    "latent_shape": list(torch.load(out_path, weights_only=True).shape)}

        t0 = time.perf_counter()
        try:
            frames = _decode_frames_pyav(video_path, self.num_frames, self.resolution)
            # frames: [T, 3, H, W]  →  encode one frame at a time to save VRAM
            latents = []
            with torch.inference_mode():
                for frame in frames:
                    x   = frame.unsqueeze(0).to(self.device)   # [1, 3, H, W]
                    lat = self.vae.encode(x).latent_dist.mode()
                    latents.append(lat.squeeze(0).cpu().half()) # [4, H//8, W//8]

            # Stack along temporal dim: [4, T, H//8, W//8]
            latent_tensor = torch.stack(latents, dim=1)

            Path(out_dir).mkdir(parents=True, exist_ok=True)
            torch.save(latent_tensor, out_path)

            return {
                "path":         video_path,
                "status":       "ok",
                "latent_shape": list(latent_tensor.shape),
                "time_s":       round(time.perf_counter() - t0, 3),
            }
        except Exception as e:
            return {"path": video_path, "status": f"failed: {e}",
                    "latent_shape": []}


def _load_passing_set(scores_path: str, motion_path: str, cq_path: str,
                       min_score: float, min_motion: float,
                       min_quality: float) -> set[str]:
    """Return set of video paths that pass all quality filters."""
    passing = None

    def intersect(s):
        nonlocal passing
        passing = s if passing is None else passing & s

    if scores_path and Path(scores_path).exists():
        with open(scores_path) as f:
            scores = json.load(f)
        intersect({r["path"] for r in scores
                   if r.get("status") == "ok" and r.get("mean_score", 0) >= min_score})

    if motion_path and Path(motion_path).exists():
        with open(motion_path) as f:
            motion = json.load(f)
        intersect({r["path"] for r in motion
                   if r.get("status") == "ok" and r.get("motion_score", 0) >= min_motion})

    if cq_path and Path(cq_path).exists():
        with open(cq_path) as f:
            cq = json.load(f)
        intersect({r["path"] for r in cq
                   if r.get("status") == "ok" and r.get("quality_score", 0) >= min_quality})

    return passing or set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir",         required=True)
    ap.add_argument("--out_dir",           default="data/latents")
    ap.add_argument("--scores",            default=None)
    ap.add_argument("--motion",            default=None)
    ap.add_argument("--captions_quality",  default=None)
    ap.add_argument("--min_score",         type=float, default=4.5)
    ap.add_argument("--min_motion",        type=float, default=3.0)
    ap.add_argument("--min_quality",       type=float, default=0.3)
    ap.add_argument("--num_gpus",          type=int,   default=1)
    ap.add_argument("--num_frames",        type=int,   default=16,
                    help="Frames to encode per video (temporal depth of latent)")
    ap.add_argument("--resolution",        type=int,   default=256,
                    help="Spatial resolution before VAE encoding (VAE downsamples by 8×)")
    args = ap.parse_args()

    all_videos = sorted(Path(args.video_dir).rglob("*.mp4"))

    # Apply quality filters before encoding — avoid encoding low-quality clips
    passing = _load_passing_set(
        args.scores, args.motion, args.captions_quality,
        args.min_score, args.min_motion, args.min_quality,
    )
    if passing:
        videos = [v for v in all_videos if str(v) in passing]
        print(f"Quality filter: {len(videos)}/{len(all_videos)} videos pass all thresholds")
    else:
        videos = all_videos
        print(f"No quality filters applied — encoding all {len(videos)} videos")

    latent_h = args.resolution // 8
    print(f"Encoding {len(videos)} videos → latents [{4}, {args.num_frames}, "
          f"{latent_h}, {latent_h}]  (float16)")
    print(f"VAE: {VAE_MODEL_ID}  |  "
          f"Actors: {args.num_gpus * ACTORS_PER_GPU} ({ACTORS_PER_GPU}/GPU)")

    wandb.init(project="video-curation", entity="rlx-labs",
               name="encode-latents", resume="allow",
               config={"vae_model":   VAE_MODEL_ID,
                       "num_frames":  args.num_frames,
                       "resolution":  args.resolution,
                       "num_gpus":    args.num_gpus,
                       "min_score":   args.min_score,
                       "min_motion":  args.min_motion,
                       "min_quality": args.min_quality})

    ray.init(num_gpus=args.num_gpus, ignore_reinit_error=True)
    n_actors = args.num_gpus * ACTORS_PER_GPU
    workers  = [LatentEncodeWorker.remote(args.resolution, args.num_frames)
                for _ in range(n_actors)]

    futures = [
        workers[i % n_actors].process.remote(str(v), args.out_dir)
        for i, v in enumerate(videos)
    ]

    results, ok, cached, failed = [], 0, 0, 0
    total   = len(videos)
    pending = list(futures)
    t0      = time.perf_counter()

    while pending:
        done, pending = ray.wait(pending, num_returns=min(32, len(pending)), timeout=60)
        for res in ray.get(done):
            results.append(res)
            if   res["status"] == "ok":     ok     += 1
            elif res["status"] == "cached": cached += 1
            else:
                failed += 1
                print(f"    ERROR: {Path(res['path']).name}: {res['status']}")
        completed = ok + cached + failed
        elapsed   = time.perf_counter() - t0
        print(f"  [{completed}/{total}]  ok={ok}  cached={cached}  "
              f"failed={failed}  elapsed={elapsed:.1f}s")
        wandb.log({"latents/ok": ok, "latents/cached": cached,
                   "latents/failed": failed, "latents/elapsed_s": elapsed})

    # Disk usage summary
    out_dir    = Path(args.out_dir)
    pt_files   = list(out_dir.glob("*.pt"))
    total_mb   = sum(p.stat().st_size for p in pt_files) / 1e6
    print(f"\nLatents written : {ok}  cached={cached}  failed={failed}")
    print(f"Disk usage      : {total_mb:.1f} MB  ({len(pt_files)} files)")
    print(f"Output dir      : {args.out_dir}")

    wandb.log({"latents/total_files": len(pt_files),
               "latents/total_mb":    total_mb})
    wandb.finish()
    ray.shutdown()


if __name__ == "__main__":
    main()
