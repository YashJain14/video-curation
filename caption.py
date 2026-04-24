"""
caption.py
----------
VLM-based captioning for video clips using Qwen2.5-VL-7B-Instruct.
Samples N keyframes per video, passes them to the VLM, generates a
single descriptive caption per clip.

Captions are stored in data/captions.json and later embedded into
WebDataset shards alongside the video bytes.

Why Qwen2.5-VL:
  - Strong video understanding, handles multi-frame input natively
  - Open weights, runs on a single A100 in 4-bit or bf16
  - Outputs natural language captions suitable for text-conditioned training

Usage:
  python caption.py --video_dir data/raw_videos --out data/captions.json \
                    --frames_per_video 4 --device cuda:0
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import ray
from PIL import Image

from decode import decode_video


MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

CAPTION_PROMPT = (
    "You are a video captioning assistant. "
    "Given several keyframes from a short video clip, write a single concise sentence "
    "describing the main action, subject, and setting. "
    "Be specific about motion and appearance. Do not mention 'frames' or 'images'."
)


def _sample_frames_pil(batches, n, is_gpu):
    """Sample n frames and return as PIL Images."""
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
def caption_video(video_path: str, frames_per_video: int = 4) -> dict:
    """
    Ray task: decode + caption one video with Qwen2.5-VL-7B.
    Uses num_gpus=1 — Qwen2.5-VL-7B in bf16 needs the full 40GB A100.
    With 4 GPUs: 4 caption tasks run in parallel (one per GPU).
    Ray sets CUDA_VISIBLE_DEVICES per task — always use cuda:0 inside.
    """
    device = "cuda:0"
    t0 = time.perf_counter()
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info

        batches, _, backend = decode_video(video_path, max_frames=64,
                                           batch_size=16, device=device)
        is_gpu = backend != "opencv_cpu"
        frames = _sample_frames_pil(batches, frames_per_video, is_gpu)
        if not frames:
            return {"path": video_path, "status": "failed: no frames", "caption": ""}

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        processor = AutoProcessor.from_pretrained(MODEL_ID)

        content = [{"type": "text", "text": CAPTION_PROMPT}]
        for img in frames:
            content.append({"type": "image", "image": img})

        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs,
            return_tensors="pt", padding=True
        ).to(device)

        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=128)
        trimmed  = out_ids[:, inputs["input_ids"].shape[1]:]
        caption  = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

        return {
            "path":    video_path,
            "status":  "ok",
            "caption": caption,
            "time_s":  time.perf_counter() - t0,
        }
    except Exception as e:
        return {"path": video_path, "status": f"failed: {e}", "caption": ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir",        required=True)
    ap.add_argument("--out",              default="data/captions.json")
    ap.add_argument("--frames_per_video", type=int, default=4)
    ap.add_argument("--num_gpus",         type=int, default=1,
                    help="Number of GPUs. Each caption task uses 1 full GPU → num_gpus tasks in parallel.")
    ap.add_argument("--ray_address",      default=None)
    args = ap.parse_args()

    videos = sorted(Path(args.video_dir).rglob("*.mp4"))
    print(f"Captioning {len(videos)} videos with {MODEL_ID} ...")
    print(f"Concurrency: {args.num_gpus} tasks (1 GPU each) across {args.num_gpus} GPU(s)")

    ray.init(num_gpus=args.num_gpus, ignore_reinit_error=True)

    futures = [
        caption_video.remote(str(v), args.frames_per_video)
        for v in videos
    ]

    results = []
    ok = failed = 0
    t0 = time.perf_counter()
    for i, res in enumerate(ray.get(futures), 1):
        results.append(res)
        if res["status"] == "ok": ok     += 1
        else:                     failed += 1
        if i % 20 == 0 or i == len(videos):
            print(f"  [{i}/{len(videos)}]  ok={ok}  failed={failed}  "
                  f"elapsed={time.perf_counter()-t0:.1f}s")
        if res["status"] == "ok":
            print(f"    {Path(res['path']).name}: {res['caption'][:80]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nCaptions saved → {out}  (ok={ok}, failed={failed})")
    ray.shutdown()


if __name__ == "__main__":
    main()
