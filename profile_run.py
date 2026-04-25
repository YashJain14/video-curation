"""
profile_run.py
--------------
GPU profiling wrapper for the embed stage using torchscope.

Runs embed under torchscope's RayProfiler to measure real GPU utilisation
across all Ray workers. Proves the 0.25 GPU/task allocation actually loads
all 4 GPUs efficiently rather than just reserving them.

torchscope does NOT need to be pip-installed. Pass --torchscope /path/to/repo
and the script adds it to sys.path before importing.

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
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
import ray


def _add_torchscope_path(ts_path: str):
    """Insert torchscope repo root into sys.path if not already there."""
    if ts_path and ts_path not in sys.path:
        sys.path.insert(0, ts_path)


def _import_torchscope(ts_path: str = ""):
    """
    Import torchscope from local repo path (no pip install needed).
    Returns (Profiler, RayProfiler, worker_profiler_context) or None.
    """
    _add_torchscope_path(ts_path)
    try:
        from torchscope import Profiler
        from torchscope.ray_profiler import RayProfiler, worker_profiler_context
        return Profiler, RayProfiler, worker_profiler_context
    except ImportError as e:
        print(f"[profile_run] torchscope not available: {e}")
        return None


# ── embed helpers (duplicated to avoid serialising module-level imports) ───────

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


# ── Ray actor ─────────────────────────────────────────────────────────────────
# Actor (not a task function) so CLIP is loaded once per worker, not once per
# video. With 10k videos a task function would reload ~400 MB of weights 10k
# times; an actor loads them once in __init__ and reuses across all calls.

@ray.remote(num_gpus=0.25)
class EmbedWorkerProfiled:
    def __init__(self, out_dir: str, frames_per_video: int,
                 ts_path: str, actor_name: str, worker_id: int, force: bool = False):
        _add_torchscope_path(ts_path)
        self.device          = "cuda:0"
        self.out_dir         = out_dir
        self.frames_per_video = frames_per_video
        self.worker_id       = worker_id
        self.ts_path         = ts_path
        self.actor_name      = actor_name

        self.force           = force
        self.model, self.processor = _load_clip(self.device)

        # Build profiling context once per actor lifetime
        self.ctx = nullcontext()
        if ts_path and actor_name:
            try:
                from torchscope.ray_profiler import RayProfiler, worker_profiler_context

                # Reconstruct a minimal RayProfiler pointing at the named aggregator.
                # Actor handles can't be pickled across the Ray task boundary, so we
                # resolve by name here inside the fresh worker process.
                rp = RayProfiler.__new__(RayProfiler)
                rp._num_workers     = 1
                rp._interval        = 0.5
                rp._report_interval = 10.0
                rp._enabled         = True
                rp._actor_handle    = ray.get_actor(actor_name)

                self.ctx = worker_profiler_context(rp, worker_id=worker_id, device=0)
                self.ctx.__enter__()
            except Exception as e:
                print(f"[worker {worker_id}] torchscope init failed: {e}")
                self.ctx = nullcontext()

    def process(self, video_path: str) -> dict:
        out_path = Path(self.out_dir) / (Path(video_path).stem + ".npz")
        if out_path.exists() and not self.force:
            return {"path": video_path, "status": "cached"}

        t0 = time.perf_counter()
        try:
            from decode import decode_video
            batches, _, backend = decode_video(
                video_path, max_frames=64, batch_size=16, device=self.device
            )
            is_gpu = backend in ("pynvvideocodec_threaded", "pynvvideocodec_simple")
            frames = _sample_frames(batches, self.frames_per_video, is_gpu)
            if not frames:
                return {"path": video_path, "status": "failed: no frames"}

            inputs = self.processor(
                images=frames, return_tensors="pt", padding=True
            ).to(self.device)
            with torch.no_grad():
                vision_out = self.model.vision_model(pixel_values=inputs["pixel_values"])
                feats      = self.model.visual_projection(vision_out.pooler_output)
                feats      = F.normalize(feats, dim=-1)

            feats_np = feats.cpu().numpy()
            mean_emb = feats_np.mean(axis=0)
            mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)

            Path(self.out_dir).mkdir(parents=True, exist_ok=True)
            np.savez(
                out_path,
                mean_embedding=mean_emb,
                frame_embeddings=feats_np,
                video_path=np.array([video_path]),
            )

            return {"path": video_path, "status": "ok",
                    "time_s": time.perf_counter() - t0}

        except Exception as e:
            return {"path": video_path, "status": f"failed: {e}"}

    def stop_profiling(self):
        try:
            self.ctx.__exit__(None, None, None)
        except Exception:
            pass


# ── driver ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir",        required=True)
    ap.add_argument("--out_dir",          default=None)
    ap.add_argument("--report_dir",       default=None)
    ap.add_argument("--frames_per_video", type=int, default=8)
    ap.add_argument("--num_gpus",         type=int, default=1)
    ap.add_argument("--torchscope",       default=None,
                    help="Path to torchscope repo root. "
                         "No pip install needed — just pass the directory.")
    ap.add_argument("--force",            action="store_true",
                    help="Ignore existing .npz cache — re-embed every video. "
                         "Required to get real GPU work for profiling.")
    ap.add_argument("--limit",            type=int, default=None,
                    help="Only process the first N videos. Use with --force "
                         "to profile a subset without re-embedding all 10k.")
    args = ap.parse_args()

    scratch    = Path(os.environ.get("SCRATCH_DIR") or
                      os.path.expanduser("~/scratch/video-curation"))
    out_dir    = Path(args.out_dir)    if args.out_dir    else scratch / "embeddings"
    report_dir = Path(args.report_dir) if args.report_dir else scratch / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    ts_path    = str(Path(args.torchscope).expanduser()) if args.torchscope else ""

    ts = _import_torchscope(ts_path)

    videos = sorted(Path(args.video_dir).rglob("*.mp4"))
    if args.limit:
        videos = videos[:args.limit]
    if not videos:
        print(f"No videos found in {args.video_dir}")
        return

    concurrency = args.num_gpus * 4
    print(f"Videos      : {len(videos)}")
    print(f"GPUs        : {args.num_gpus}  (0.25/task → {concurrency} concurrent)")
    print(f"Torchscope  : {'enabled  path=' + ts_path if ts else 'disabled'}")

    ray.init(num_gpus=args.num_gpus, ignore_reinit_error=True)

    # ── torchscope setup ──────────────────────────────────────────────────────
    rp          = None
    actor_name  = ""
    driver_prof = None

    if ts:
        Profiler, RayProfiler, _ = ts

        # Create aggregator actor with a fixed name so workers can find it.
        # We reach into RayProfiler to pre-create the named actor before
        # dispatching tasks — workers call ray.get_actor(actor_name) instead
        # of receiving an un-picklable handle.
        ACTOR_NAME = "torchscope_aggregator"
        from torchscope.ray_profiler import _AggregatorActor
        named_actor = _AggregatorActor.options(
            name=ACTOR_NAME, lifetime="detached"
        ).remote(concurrency)

        rp = RayProfiler(num_workers=concurrency)
        rp._actor_handle = named_actor   # point driver's RayProfiler at named actor
        actor_name = ACTOR_NAME

        driver_prof = Profiler(
            interval    = 0.5,
            gpu_ids     = list(range(args.num_gpus)),
            job_name    = "video_curation_embed",
            export_json = str(report_dir / "embed_gpu_profile.json"),
        )
        driver_prof.start()

    # ── dispatch tasks ────────────────────────────────────────────────────────
    workers = [
        EmbedWorkerProfiled.remote(
            str(out_dir), args.frames_per_video,
            ts_path, actor_name, i, args.force,
        )
        for i in range(concurrency)
    ]

    t0 = time.perf_counter()
    futures = [
        workers[i % concurrency].process.remote(str(v))
        for i, v in enumerate(videos)
    ]

    ok = failed = cached = 0
    total   = len(videos)
    pending = list(futures)
    while pending:
        done, pending = ray.wait(pending, num_returns=min(50, len(pending)), timeout=60)
        for res in ray.get(done):
            s = res.get("status", "")
            if   s == "ok":     ok     += 1
            elif s == "cached": cached += 1
            else:
                failed += 1
                print(f"    ERROR: {Path(res['path']).name}: {s}")
        completed = ok + failed + cached
        if completed % 50 == 0 or completed == total:
            print(f"  [{completed}/{total}]  ok={ok}  cached={cached}  "
                  f"failed={failed}  {time.perf_counter()-t0:.1f}s")

    # Stop torchscope profiling inside each actor before collecting reports
    ray.get([w.stop_profiling.remote() for w in workers])

    elapsed = time.perf_counter() - t0
    print(f"\nDone: {len(videos)} videos in {elapsed:.1f}s  "
          f"({len(videos)/elapsed:.1f} videos/s)")

    # ── generate reports ──────────────────────────────────────────────────────
    if ts and rp:
        print("\nGenerating cluster report ...")
        analysis = rp.aggregate_report(
            output_path = str(report_dir / "embed_cluster_profile.html"),
            title       = "Video Curation — Embed Stage (4-GPU Ray cluster)",
        )
        cs = analysis.get("_cluster_stats", {})
        if cs:
            print(f"  Avg GPU util     : {cs.get('cluster_avg_gpu_util', '?')}%")
            print(f"  Bottleneck worker: {cs.get('bottleneck_worker_id', '?')}  "
                  f"({cs.get('bottleneck_avg_util', '?')}%)")
            print(f"  Imbalance        : {cs.get('imbalance_pct', '?')}%")

    if ts and driver_prof:
        driver_prof.report(
            output_path   = str(report_dir / "embed_driver_profile.html"),
            title         = "Video Curation — Driver GPU Profile",
            custom_stages = {"embed_per_video": elapsed / max(len(videos), 1)},
            metadata      = {
                "videos":      len(videos),
                "num_gpus":    args.num_gpus,
                "concurrency": concurrency,
                "throughput":  f"{len(videos)/elapsed:.1f} videos/s",
                "ok/failed":   f"{ok}/{failed}",
            },
        )
        print(f"Reports → {report_dir}/")

    # Kill the detached actor so it doesn't persist between runs
    if actor_name:
        try:
            ray.kill(ray.get_actor(actor_name))
        except Exception:
            pass

    ray.shutdown()


if __name__ == "__main__":
    main()
