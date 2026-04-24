"""
profile_run.py
--------------
GPU profiling wrapper for the embed stage using torchscope.

Runs embed under torchscope's RayProfiler to measure real GPU utilisation
across all Ray workers. Proves the 0.25 GPU/task allocation actually loads
all 4 GPUs efficiently rather than just reserving them.

Setup (run once on cluster after cloning torchscope):
    pip install -e /path/to/torchscope

Output:
    $SCRATCH_DIR/reports/embed_cluster_profile.html  — per-worker + cluster report
    $SCRATCH_DIR/reports/embed_driver_profile.html   — driver-side GPU timeline

Usage:
    python profile_run.py \
        --video_dir    $SCRATCH_DIR/raw_videos \
        --out_dir      $SCRATCH_DIR/embeddings \
        --num_gpus     4 \
        --torchscope   /home/users/ntu/yash012/torchscope
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import ray


def _import_torchscope(ts_path: str = None):
    """
    Import torchscope from an editable install or a local path.
    Returns (Profiler, RayProfiler, worker_profiler_context) or None if unavailable.
    """
    if ts_path:
        sys.path.insert(0, str(ts_path))
    try:
        from torchscope import Profiler
        from torchscope.ray_profiler import RayProfiler, worker_profiler_context
        return Profiler, RayProfiler, worker_profiler_context
    except ImportError as e:
        print(f"torchscope not available: {e}")
        print("Install with: pip install -e /path/to/torchscope")
        return None


# ── embed logic (duplicated here to avoid Ray serialisation of module refs) ───

def _load_clip(device):
    from transformers import CLIPProcessor, CLIPModel
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor


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


# ── Ray task ──────────────────────────────────────────────────────────────────

@ray.remote(num_gpus=0.25)
def embed_video_profiled(video_path: str, out_dir: str,
                         frames_per_video: int,
                         rp,            # RayProfiler instance (or None)
                         worker_id: int,
                         ts_path: str) -> dict:
    """
    Embed one video, wrapped in torchscope worker profiling.
    rp is the RayProfiler driver handle; each task pushes its GPU summary back.
    """
    # Import torchscope inside the task (Ray workers are fresh processes)
    ts = _import_torchscope(ts_path)

    device   = "cuda:0"
    out_path = Path(out_dir) / (Path(video_path).stem + ".npz")
    if out_path.exists():
        return {"path": video_path, "status": "cached"}

    # Set up context manager: torchscope if available, else plain nullcontext
    if ts and rp is not None:
        _, _, worker_profiler_context = ts
        ctx = worker_profiler_context(rp, worker_id=worker_id, device=0)
    else:
        from contextlib import nullcontext
        ctx = nullcontext()

    t0 = time.perf_counter()
    try:
        from decode import decode_video
        with ctx:
            batches, _, backend = decode_video(video_path, max_frames=64,
                                               batch_size=16, device=device)
            is_gpu = backend != "opencv_cpu"
            frames = _sample_frames(batches, frames_per_video, is_gpu)
            if not frames:
                return {"path": video_path, "status": "failed: no frames"}

            model, processor = _load_clip(device)
            inputs = processor(images=frames, return_tensors="pt",
                               padding=True).to(device)
            with torch.no_grad():
                feats = model.get_image_features(**inputs)
                feats = F.normalize(feats, dim=-1)

        feats_np = feats.cpu().numpy()
        mean_emb = feats_np.mean(axis=0)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        np.savez(out_path,
                 mean_embedding=mean_emb,
                 frame_embeddings=feats_np,
                 video_path=np.array([video_path]))

        return {"path": video_path, "status": "ok",
                "time_s": time.perf_counter() - t0}

    except Exception as e:
        return {"path": video_path, "status": f"failed: {e}"}


# ── driver ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir",        required=True)
    ap.add_argument("--out_dir",          default=None)
    ap.add_argument("--report_dir",       default=None)
    ap.add_argument("--frames_per_video", type=int, default=8)
    ap.add_argument("--num_gpus",         type=int, default=1)
    ap.add_argument("--torchscope",       default=None,
                    help="Path to torchscope repo, e.g. ~/torchscope. "
                         "Not needed if already installed with pip install -e.")
    args = ap.parse_args()

    scratch    = Path(os.environ.get("SCRATCH_DIR") or
                      os.path.expanduser("~/scratch/video-curation"))
    out_dir    = Path(args.out_dir)    if args.out_dir    else scratch / "embeddings"
    report_dir = Path(args.report_dir) if args.report_dir else scratch / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    ts_path    = str(Path(args.torchscope).expanduser()) if args.torchscope else ""

    ts = _import_torchscope(ts_path or None)

    videos = sorted(Path(args.video_dir).rglob("*.mp4"))
    if not videos:
        print(f"No videos found in {args.video_dir}")
        return

    concurrency = args.num_gpus * 4
    print(f"Videos      : {len(videos)}")
    print(f"GPUs        : {args.num_gpus}  (0.25/task → {concurrency} concurrent)")
    print(f"Torchscope  : {'enabled' if ts else 'disabled (no profiling)'}")

    ray.init(num_gpus=args.num_gpus, ignore_reinit_error=True)

    # Driver-side objects
    rp          = None
    driver_prof = None
    if ts:
        Profiler, RayProfiler, _ = ts
        rp = RayProfiler(num_workers=concurrency)
        driver_prof = Profiler(
            interval    = 0.5,
            gpu_ids     = list(range(args.num_gpus)),
            job_name    = "video_curation_embed",
            export_json = str(report_dir / "embed_gpu_profile.json"),
        )
        driver_prof.start()

    t0 = time.perf_counter()
    futures = [
        embed_video_profiled.remote(
            str(v), str(out_dir), args.frames_per_video,
            rp, i % concurrency, ts_path
        )
        for i, v in enumerate(videos)
    ]

    ok = failed = cached = 0
    for i, res in enumerate(ray.get(futures), 1):
        s = res.get("status", "")
        if   s == "ok":     ok     += 1
        elif s == "cached": cached += 1
        else:               failed += 1
        if i % 50 == 0 or i == len(videos):
            print(f"  [{i}/{len(videos)}]  ok={ok}  cached={cached}  "
                  f"failed={failed}  {time.perf_counter()-t0:.1f}s")

    elapsed = time.perf_counter() - t0
    print(f"\nDone: {len(videos)} videos in {elapsed:.1f}s  "
          f"({len(videos)/elapsed:.1f} videos/s)")

    # ── reports ───────────────────────────────────────────────────────────────
    if ts and rp:
        print("\nGenerating cluster report ...")
        analysis = rp.aggregate_report(
            output_path = str(report_dir / "embed_cluster_profile.html"),
            title       = "Video Curation — Embed Stage (4-GPU Ray cluster)",
        )
        cs = analysis.get("_cluster_stats", {})
        if cs:
            print(f"  Avg GPU util     : {cs.get('cluster_avg_gpu_util','?')}%")
            print(f"  Bottleneck worker: {cs.get('bottleneck_worker_id','?')}  "
                  f"({cs.get('bottleneck_avg_util','?')}%)")
            print(f"  Imbalance        : {cs.get('imbalance_pct','?')}%")

    if ts and driver_prof:
        driver_prof.report(
            output_path   = str(report_dir / "embed_driver_profile.html"),
            title         = "Video Curation — Driver GPU Profile",
            custom_stages = {"embed_per_video": elapsed / max(len(videos), 1)},
            metadata      = {
                "videos":       len(videos),
                "num_gpus":     args.num_gpus,
                "concurrency":  concurrency,
                "throughput":   f"{len(videos)/elapsed:.1f} videos/s",
                "ok/failed":    f"{ok}/{failed}",
            },
        )
        print(f"Reports → {report_dir}/")

    ray.shutdown()


if __name__ == "__main__":
    main()
