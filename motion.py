"""
motion.py
---------
Motion quality scoring for video clips.

Two complementary signals per video:
  - flow_magnitude  : mean optical flow magnitude across sampled frame pairs
                      (low → static / nearly frozen; high → fast/chaotic motion)
  - flow_variance   : variance of per-pair magnitudes
                      (high → unsteady / shaky camera)
  - ssim_mean       : mean structural similarity between consecutive sampled frames
                      (low → scene cuts, heavy motion blur, or flicker;
                       high → temporally coherent, smooth clip)
  - motion_score    : composite [0–10] combining flow magnitude and temporal
                      coherence; higher = smoother motion, better for training

Why these signals matter for video diffusion training:
  - Aesthetic score measures individual frame quality — a sharp static frame
    scores well even if the clip is frozen or flickering.
  - Diffusion models learn temporal consistency from the training data.
    Clips with erratic motion, heavy motion blur, or scene cuts inside a
    10-second clip are actively harmful: they teach the model that frames
    within a clip can be unrelated.
  - flow_variance catches shaky handheld footage that aesthetic score misses.
  - ssim_mean catches motion blur and scene cuts that optical flow alone misses.

Output: motion_scores.json [{path, flow_magnitude, flow_variance, ssim_mean,
                              motion_score, status}]
Cache:  motion_cache/<stem>.motion.json   (skipped on rerun)

Ray concurrency:
  CPU-only stage — no GPU needed. Using num_cpus=2 per actor for parallelism
  across the node's CPU cores. Adjust ACTORS to match your node.

Usage:
  python motion.py --video_dir $SCRATCH_DIR/raw_videos \\
                   --out $SCRATCH_DIR/motion_scores.json \\
                   --num_workers 16
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import ray
import wandb


FRAMES_TO_SAMPLE = 8   # frame pairs to analyse per video


def _sample_frames_pyav(video_path: str, n: int) -> list:
    """Sample n evenly-spaced frames as RGB numpy arrays via PyAV."""
    import av
    frames = []
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        total  = stream.frames or 0
        # Seek to evenly spaced pts if we know total frames; otherwise decode all
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        return []
    indices = np.linspace(0, len(frames) - 1, min(n + 1, len(frames)), dtype=int)
    return [frames[i] for i in indices]


def _optical_flow_magnitude(f1: np.ndarray, f2: np.ndarray) -> float:
    """
    Dense optical flow magnitude between two RGB frames using Farneback.
    Returns mean flow vector magnitude in pixels.
    """
    import cv2
    g1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        g1, g2,
        None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(mag.mean())


def _ssim(f1: np.ndarray, f2: np.ndarray) -> float:
    """
    Structural similarity between two RGB frames.
    Converts to grayscale, computes SSIM via sliding window.
    Returns scalar in [0, 1]; 1.0 = identical frames.
    """
    import cv2
    g1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY).astype(np.float32)

    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1, mu2 = g1.mean(), g2.mean()
    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sig1_sq = g1.var()
    sig2_sq = g2.var()
    sig12   = float(np.mean((g1 - mu1) * (g2 - mu2)))

    ssim = ((2 * mu12 + C1) * (2 * sig12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sig1_sq + sig2_sq + C2)
    )
    return float(np.clip(ssim, 0.0, 1.0))


def _motion_score(flow_mag: float, flow_var: float, ssim_mean: float) -> float:
    """
    Composite motion quality score in [0, 10].

    Components:
      - coherence   (0–5): based on ssim_mean — rewards temporal smoothness
      - flow_reward (0–3): rewards moderate motion (not frozen, not chaotic)
      - stability   (0–2): penalises high flow variance (camera shake)

    Design targets:
      - Smooth, moderately moving clip  → ~8–9
      - Slow pan / nearly static        → ~5–6  (ssim high, flow low)
      - Shaky handheld action           → ~3–5  (flow_var high)
      - Scene cut inside clip           → ~1–3  (ssim drops sharply)
    """
    coherence   = ssim_mean * 5.0
    # sigmoid-like reward: peaks around flow_mag=3–8 px, falls off at extremes
    flow_reward = 3.0 * np.exp(-((flow_mag - 5.0) ** 2) / 50.0)
    # penalise high variance: variance > 10 px² is shaky
    stability   = 2.0 * np.exp(-flow_var / 10.0)
    return float(np.clip(coherence + flow_reward + stability, 0.0, 10.0))


@ray.remote(num_cpus=2)
class MotionWorker:
    def process(self, video_path: str, cache_dir: str) -> dict:
        cache_path = Path(cache_dir) / (Path(video_path).stem + ".motion.json")
        if cache_path.exists():
            return json.loads(cache_path.read_text())

        t0 = time.perf_counter()
        try:
            frames = _sample_frames_pyav(video_path, FRAMES_TO_SAMPLE)
            if len(frames) < 2:
                return {"path": video_path, "status": "failed: too few frames",
                        "flow_magnitude": 0.0, "flow_variance": 0.0,
                        "ssim_mean": 0.0, "motion_score": 0.0}

            flow_mags, ssims = [], []
            for i in range(len(frames) - 1):
                flow_mags.append(_optical_flow_magnitude(frames[i], frames[i + 1]))
                ssims.append(_ssim(frames[i], frames[i + 1]))

            flow_mag = float(np.mean(flow_mags))
            flow_var = float(np.var(flow_mags))
            ssim_m   = float(np.mean(ssims))
            m_score  = _motion_score(flow_mag, flow_var, ssim_m)

            result = {
                "path":           video_path,
                "status":         "ok",
                "flow_magnitude": round(flow_mag, 4),
                "flow_variance":  round(flow_var, 4),
                "ssim_mean":      round(ssim_m, 4),
                "motion_score":   round(m_score, 4),
                "time_s":         round(time.perf_counter() - t0, 3),
            }
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(result))
            return result

        except Exception as e:
            return {"path": video_path, "status": f"failed: {e}",
                    "flow_magnitude": 0.0, "flow_variance": 0.0,
                    "ssim_mean": 0.0, "motion_score": 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir",   required=True)
    ap.add_argument("--out",         default="data/motion_scores.json")
    ap.add_argument("--num_workers", type=int, default=16,
                    help="CPU Ray actors (no GPU needed)")
    ap.add_argument("--min_motion",  type=float, default=3.0,
                    help="Motion score threshold for downstream filtering")
    args = ap.parse_args()

    videos = sorted(Path(args.video_dir).rglob("*.mp4"))
    print(f"Motion scoring {len(videos)} videos  ({args.num_workers} workers) ...")

    wandb.init(project="video-curation", entity="rlx-labs",
               name="motion", resume="allow",
               config={"num_workers": args.num_workers,
                       "min_motion":  args.min_motion,
                       "frames":      FRAMES_TO_SAMPLE})

    ray.init(ignore_reinit_error=True)
    workers   = [MotionWorker.remote() for _ in range(args.num_workers)]
    cache_dir = str(Path(args.out).parent / "motion_cache")

    futures = [
        workers[i % args.num_workers].process.remote(str(v), cache_dir)
        for i, v in enumerate(videos)
    ]

    results, ok, failed = [], 0, 0
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
        wandb.log({"motion/ok": ok, "motion/failed": failed,
                   "motion/completed": completed, "motion/elapsed_s": elapsed})

    # Summary stats
    ok_results  = [r for r in results if r["status"] == "ok"]
    scores      = [r["motion_score"]  for r in ok_results]
    flow_mags   = [r["flow_magnitude"] for r in ok_results]
    ssims       = [r["ssim_mean"]      for r in ok_results]
    passing     = sum(1 for s in scores if s >= args.min_motion)

    print(f"\nMotion score stats:")
    print(f"  Mean  : {np.mean(scores):.2f}")
    print(f"  Median: {np.median(scores):.2f}")
    print(f"  ≥{args.min_motion}: {passing}/{len(scores)} ({100*passing/max(len(scores),1):.1f}%)")
    print(f"  Flow magnitude mean : {np.mean(flow_mags):.2f} px")
    print(f"  SSIM mean           : {np.mean(ssims):.3f}")

    wandb.log({
        "motion/score_mean":       float(np.mean(scores)),
        "motion/score_median":     float(np.median(scores)),
        "motion/flow_mag_mean":    float(np.mean(flow_mags)),
        "motion/ssim_mean":        float(np.mean(ssims)),
        "motion/passing":          passing,
        "motion/passing_pct":      100 * passing / max(len(scores), 1),
        "motion/score_hist":       wandb.Histogram(scores),
        "motion/flow_hist":        wandb.Histogram(flow_mags),
        "motion/ssim_hist":        wandb.Histogram(ssims),
    })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMotion scores saved → {out}")
    wandb.finish()
    ray.shutdown()


if __name__ == "__main__":
    main()
