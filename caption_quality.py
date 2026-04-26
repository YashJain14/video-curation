"""
caption_quality.py
------------------
Caption quality analysis for VLM-generated video captions.

Why caption quality matters for video diffusion training:
  Text-to-video diffusion models learn to associate text conditioning with
  visual content during training. If the caption for a clip is vague
  ("a person is doing something"), the model learns a weak association and
  text conditioning becomes less effective. High-quality captions that
  precisely describe motion, subject, and setting produce models that better
  follow text prompts at inference time.

Three complementary signals:

  1. clip_alignment (0–1):
     Cosine similarity between the CLIP ViT-B/32 text embedding of the caption
     and the CLIP image embedding of the video (already computed by embed.py).
     High alignment means the caption accurately describes what is visually
     present. Low alignment means the VLM hallucinated or produced an
     off-topic description.
     Typical range: 0.15–0.40 for Kinetics clips.
     Threshold: 0.20 (captions below this are likely misaligned).

  2. caption_length (tokens):
     Number of whitespace-delimited tokens in the caption. Very short captions
     (<5 tokens: "a person running") carry too little conditioning signal.
     Very long captions (>60 tokens) often contain repetition or hallucination.
     Target range: 8–40 tokens.

  3. specificity_score (0–1):
     Heuristic: ratio of non-stopword tokens to total tokens. A caption like
     "a man is doing something in a place" scores low; "a skateboarder executes
     a kickflip on a concrete half-pipe" scores high. Specificity correlates
     with useful conditioning signal.

Output:
  caption_quality.json — [{path, caption, clip_alignment, caption_length,
                            specificity_score, quality_score, status}]

Cache: caption_quality_cache/<stem>.cq.json

Usage:
  python caption_quality.py \\
    --captions  $SCRATCH_DIR/captions.json \\
    --emb_dir   $SCRATCH_DIR/embeddings \\
    --out       $SCRATCH_DIR/caption_quality.json \\
    --num_gpus  1
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import ray
import wandb
from transformers import CLIPProcessor, CLIPModel


MODEL_ID    = "openai/clip-vit-base-patch32"
ACTORS_PER_GPU = 2

# Common English stopwords — kept minimal to avoid nltk dependency
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "through", "during",
    "and", "or", "but", "not", "no", "so", "this", "that", "these", "those",
    "it", "its", "he", "she", "they", "we", "you", "i", "my", "his", "her",
    "their", "our", "your", "its", "who", "which", "what", "where", "when",
    "there", "here", "some", "each", "very", "just", "as", "if",
}


def _specificity(caption: str) -> float:
    tokens = caption.lower().split()
    if not tokens:
        return 0.0
    content = [t for t in tokens if t.strip(".,!?;:\"'") not in _STOPWORDS]
    return len(content) / len(tokens)


def _quality_score(clip_alignment: float, length: int, specificity: float) -> float:
    """
    Composite caption quality score in [0, 1].

    Weights reflect importance for diffusion conditioning:
      clip_alignment (50%) — most important: is it accurate?
      specificity    (30%) — is it descriptive?
      length penalty (20%) — is it in the useful range?

    Length scoring: peaks at 15–30 tokens, penalises very short/long.
    """
    # Normalise alignment from [0.10, 0.40] → [0, 1]
    align_norm = float(np.clip((clip_alignment - 0.10) / 0.30, 0.0, 1.0))

    # Length score: triangle peak at 20 tokens
    if length < 5:
        len_score = length / 5.0 * 0.5       # short captions: up to 0.5
    elif length <= 30:
        len_score = 0.5 + (length - 5) / 50.0
    else:
        len_score = max(0.0, 1.0 - (length - 30) / 60.0)

    return float(np.clip(
        0.50 * align_norm + 0.30 * specificity + 0.20 * len_score,
        0.0, 1.0,
    ))


@ray.remote(num_gpus=1.0 / ACTORS_PER_GPU)
class CaptionQualityWorker:
    def __init__(self):
        self.device = "cuda:0"
        self.model     = CLIPModel.from_pretrained(MODEL_ID).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(MODEL_ID)
        self.model.eval()

    def _encode_text(self, caption: str) -> np.ndarray:
        inputs = self.processor(text=[caption], return_tensors="pt",
                                padding=True, truncation=True).to(self.device)
        with torch.inference_mode():
            out  = self.model.text_model(**{k: v for k, v in inputs.items()
                                            if k in ("input_ids", "attention_mask")})
            feat = self.model.text_projection(out.pooler_output)
            feat = F.normalize(feat, dim=-1)
        return feat.cpu().numpy()[0]

    def process(self, video_path: str, caption: str,
                emb_dir: str, cache_dir: str) -> dict:
        cache_path = Path(cache_dir) / (Path(video_path).stem + ".cq.json")
        if cache_path.exists():
            return json.loads(cache_path.read_text())

        t0 = time.perf_counter()
        try:
            # Load precomputed video embedding from embed.py
            stem    = Path(video_path).stem
            npz     = Path(emb_dir) / f"{stem}.npz"
            if not npz.exists():
                return {"path": video_path, "status": "failed: no embedding",
                        "clip_alignment": 0.0, "caption_length": 0,
                        "specificity_score": 0.0, "quality_score": 0.0,
                        "caption": caption}

            vid_emb = np.load(npz)["mean_embedding"].astype(np.float32)
            vid_emb = vid_emb / (np.linalg.norm(vid_emb) + 1e-8)

            txt_emb      = self._encode_text(caption)
            alignment    = float(np.dot(vid_emb, txt_emb))
            length       = len(caption.split())
            specificity  = _specificity(caption)
            quality      = _quality_score(alignment, length, specificity)

            result = {
                "path":              video_path,
                "status":            "ok",
                "caption":           caption,
                "clip_alignment":    round(alignment,   4),
                "caption_length":    length,
                "specificity_score": round(specificity, 4),
                "quality_score":     round(quality,     4),
                "time_s":            round(time.perf_counter() - t0, 3),
            }
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(result))
            return result

        except Exception as e:
            return {"path": video_path, "status": f"failed: {e}",
                    "clip_alignment": 0.0, "caption_length": 0,
                    "specificity_score": 0.0, "quality_score": 0.0,
                    "caption": caption}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions",     required=True,
                    help="captions.json produced by caption.py")
    ap.add_argument("--emb_dir",      required=True,
                    help="Directory of .npz embeddings from embed.py")
    ap.add_argument("--out",          default="data/caption_quality.json")
    ap.add_argument("--num_gpus",     type=int, default=1)
    ap.add_argument("--min_quality",  type=float, default=0.65,
                    help="Quality score threshold for downstream shard filtering")
    ap.add_argument("--min_alignment",type=float, default=0.20,
                    help="CLIP alignment threshold")
    args = ap.parse_args()

    with open(args.captions) as f:
        caption_records = json.load(f)

    valid = [(r["path"], r["caption"])
             for r in caption_records
             if r.get("status") == "ok" and r.get("caption")]
    print(f"Scoring caption quality for {len(valid)} videos ...")

    wandb.init(project="video-curation", entity="rlx-labs",
               name="caption-quality", resume="allow",
               config={"num_gpus":      args.num_gpus,
                       "min_quality":   args.min_quality,
                       "min_alignment": args.min_alignment})

    ray.init(ignore_reinit_error=True)
    n_actors = args.num_gpus * ACTORS_PER_GPU
    workers  = [CaptionQualityWorker.remote() for _ in range(n_actors)]
    cache_dir = str(Path(args.out).parent / "caption_quality_cache")

    futures = [
        workers[i % n_actors].process.remote(path, caption, args.emb_dir, cache_dir)
        for i, (path, caption) in enumerate(valid)
    ]

    results, ok, failed = [], 0, 0
    total   = len(valid)
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
        wandb.log({"cq/ok": ok, "cq/failed": failed,
                   "cq/completed": completed, "cq/elapsed_s": elapsed})

    ok_results   = [r for r in results if r["status"] == "ok"]
    alignments   = [r["clip_alignment"]    for r in ok_results]
    lengths      = [r["caption_length"]    for r in ok_results]
    specificities= [r["specificity_score"] for r in ok_results]
    qualities    = [r["quality_score"]     for r in ok_results]

    passing_quality   = sum(1 for q in qualities   if q >= args.min_quality)
    passing_alignment = sum(1 for a in alignments  if a >= args.min_alignment)

    print(f"\nCaption quality stats:")
    print(f"  CLIP alignment  — mean={np.mean(alignments):.3f}  "
          f"median={np.median(alignments):.3f}  "
          f"≥{args.min_alignment}: {passing_alignment}/{len(alignments)} "
          f"({100*passing_alignment/max(len(alignments),1):.1f}%)")
    print(f"  Caption length  — mean={np.mean(lengths):.1f}  "
          f"median={np.median(lengths):.1f}  tokens")
    print(f"  Specificity     — mean={np.mean(specificities):.3f}")
    print(f"  Quality score   — mean={np.mean(qualities):.3f}  "
          f"≥{args.min_quality}: {passing_quality}/{len(qualities)} "
          f"({100*passing_quality/max(len(qualities),1):.1f}%)")

    wandb.log({
        "cq/alignment_mean":     float(np.mean(alignments)),
        "cq/alignment_median":   float(np.median(alignments)),
        "cq/length_mean":        float(np.mean(lengths)),
        "cq/specificity_mean":   float(np.mean(specificities)),
        "cq/quality_mean":       float(np.mean(qualities)),
        "cq/passing_quality":    passing_quality,
        "cq/passing_alignment":  passing_alignment,
        "cq/alignment_hist":     wandb.Histogram(alignments),
        "cq/length_hist":        wandb.Histogram(lengths),
        "cq/quality_hist":       wandb.Histogram(qualities),
    })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nCaption quality saved → {out}")
    wandb.finish()
    ray.shutdown()


if __name__ == "__main__":
    main()
