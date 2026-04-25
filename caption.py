"""
caption.py
----------
VLM-based captioning for video clips using Qwen3-VL-8B-Instruct.
Samples N keyframes per video, passes them to the VLM, generates a
single descriptive caption per clip.

Captions are stored in data/captions.json and later embedded into
WebDataset shards alongside the video bytes.

Why Qwen3-VL:
  - Strong video understanding, handles multi-frame input natively
  - Open weights, runs on a single A100 in 4-bit or bf16
  - Outputs natural language captions suitable for text-conditioned training

Usage:
  python caption.py --video_dir data/raw_videos --out data/captions.json \
                    --frames_per_video 4 --num_gpus 4
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import ray
import wandb
from PIL import Image

from decode import decode_video


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

CAPTION_PROMPT = (
    "You are a video captioning assistant. "
    "Given several keyframes from a short video clip, write a single concise sentence "
    "describing the main action, subject, and setting. "
    "Be specific about motion and appearance. Do not mention 'frames' or 'images'."
)


def _sample_frames_pil(batches, n, is_gpu):
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
    return [Image.fromarray(all_frames[i]) for i in indices]


@ray.remote(num_gpus=1)
class CaptionWorker:
    def __init__(self):
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID, dtype="auto", device_map="auto"
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)

    def process(self, video_path: str, frames_per_video: int, cache_dir: str) -> dict:
        import json
        cache_path = Path(cache_dir) / (Path(video_path).stem + ".caption.json")
        if cache_path.exists():
            return json.loads(cache_path.read_text())

        t0 = time.perf_counter()
        try:
            batches, _, backend = decode_video(video_path, max_frames=64,
                                               batch_size=16, device="cuda")
            is_gpu = backend != "opencv_cpu"
            frames = _sample_frames_pil(batches, frames_per_video, is_gpu)
            if not frames:
                return {"path": video_path, "status": "failed: no frames", "caption": ""}

            content = [{"type": "text", "text": CAPTION_PROMPT}]
            for img in frames:
                content.append({"type": "image", "image": img})

            messages = [{"role": "user", "content": content}]
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt"
            ).to(self.model.device)

            with torch.inference_mode():
                out_ids = self.model.generate(**inputs, max_new_tokens=128)
            trimmed = [
                out[len(in_ids):] for in_ids, out in zip(inputs["input_ids"], out_ids)
            ]
            caption = self.processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()

            result = {
                "path":    video_path,
                "status":  "ok",
                "caption": caption,
                "time_s":  time.perf_counter() - t0,
            }
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(result))
            return result
        except Exception as e:
            return {"path": video_path, "status": f"failed: {e}", "caption": ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir",        required=True)
    ap.add_argument("--out",              default="data/captions.json")
    ap.add_argument("--frames_per_video", type=int, default=4)
    ap.add_argument("--num_gpus",         type=int, default=1,
                    help="One CaptionWorker per GPU — num_gpus tasks in parallel.")
    ap.add_argument("--ray_address",      default=None)
    args = ap.parse_args()

    videos = sorted(Path(args.video_dir).rglob("*.mp4"))
    print(f"Captioning {len(videos)} videos with {MODEL_ID} ...")
    print(f"Actors: {args.num_gpus} (1 per GPU)")

    wandb.init(project="video-curation", entity="rlx-labs",
               name="caption", resume="allow", id="caption-stage")

    ray.init(num_gpus=args.num_gpus, ignore_reinit_error=True)

    workers = [CaptionWorker.remote() for _ in range(args.num_gpus)]

    cache_dir = str(Path(args.out).parent / "caption_cache")
    futures = [
        workers[i % args.num_gpus].process.remote(str(v), args.frames_per_video, cache_dir)
        for i, v in enumerate(videos)
    ]

    results = []
    ok = failed = 0
    total   = len(videos)
    pending = list(futures)
    t0      = time.perf_counter()

    while pending:
        done, pending = ray.wait(pending, num_returns=min(20, len(pending)), timeout=60)
        for res in ray.get(done):
            results.append(res)
            if res["status"] == "ok":
                ok += 1
                print(f"    {Path(res['path']).name}: {res['caption'][:80]}")
            else:
                failed += 1
                print(f"    ERROR: {Path(res['path']).name}: {res['status']}")
        completed = ok + failed
        elapsed   = time.perf_counter() - t0
        print(f"  [{completed}/{total}]  ok={ok}  failed={failed}  elapsed={elapsed:.1f}s")
        wandb.log({"caption/ok": ok, "caption/failed": failed,
                   "caption/completed": completed, "caption/total": total,
                   "caption/elapsed_s": elapsed})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nCaptions saved → {out}  (ok={ok}, failed={failed})")
    wandb.finish()
    ray.shutdown()


if __name__ == "__main__":
    main()
