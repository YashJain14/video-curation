"""
filter.py
---------
CLIP-based semantic filtering of video clips.

Given a text query (e.g. "outdoor sports", "person playing instrument"),
embed it with CLIP and keep only videos whose mean embedding has
cosine similarity >= threshold against the query.

This is how you semantically select a data subset without manual labelling:
  - "Keep only high-motion outdoor clips" → query = "outdoor action sport"
  - "Remove static scenes" → query = "static empty room" → invert filter
  - "Select cooking videos" → query = "person cooking food in kitchen"

Design:
  - Loads precomputed embeddings from data/embeddings/*.npz (from embed.py)
  - Encodes text query with CLIP ViT-B/32
  - Computes cosine similarity (embeddings already L2-normalised → dot product)
  - Writes kept/removed video paths to data/filter_results.json

Usage:
  python filter.py --query "outdoor sports action" --threshold 0.2
  python filter.py --query "person playing musical instrument" --threshold 0.22
  python filter.py --query "cooking food kitchen" --threshold 0.2 --invert
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel

MODEL_ID = "openai/clip-vit-base-patch32"


def load_embeddings(emb_dir: Path) -> tuple[list[str], np.ndarray]:
    paths   = sorted(emb_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No embeddings found in {emb_dir}. Run embed.py first.")
    video_paths, vectors = [], []
    for p in paths:
        data = np.load(p, allow_pickle=True)
        video_paths.append(str(data["video_path"][0]))
        vectors.append(data["mean_embedding"].astype(np.float32))
    embeddings = np.stack(vectors)                                    # [N, 512]
    norms      = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)                         # re-normalise
    return video_paths, embeddings


def encode_text(query: str, device: str) -> np.ndarray:
    model     = CLIPModel.from_pretrained(MODEL_ID).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model.eval()
    inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_out = model.text_model(input_ids=inputs["input_ids"],
                                    attention_mask=inputs["attention_mask"])
        feats = model.text_projection(text_out.pooler_output)         # [1, 512]
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()[0]                                     # [512]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query",      required=True,
                    help="Text query, e.g. 'outdoor sports action'")
    ap.add_argument("--emb_dir",    default="data/embeddings")
    ap.add_argument("--threshold",  type=float, default=0.2,
                    help="Cosine similarity threshold (CLIP ViT-B/32 typical range 0.15–0.35)")
    ap.add_argument("--invert",     action="store_true",
                    help="Keep videos BELOW threshold (exclude matching content)")
    ap.add_argument("--out",        default="data/filter_results.json")
    ap.add_argument("--device",     default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    t0 = time.perf_counter()

    print(f"Loading embeddings from {args.emb_dir} ...")
    video_paths, embeddings = load_embeddings(Path(args.emb_dir))
    print(f"  {len(video_paths)} videos  dim={embeddings.shape[1]}")

    print(f"Encoding query: \"{args.query}\" ...")
    text_emb = encode_text(args.query, args.device)                   # [512]

    # Cosine similarity: embeddings are L2-normalised → dot product
    sims = (embeddings @ text_emb).tolist()                           # [N]

    kept = []
    removed = []
    for path, sim in zip(video_paths, sims):
        passes = sim >= args.threshold
        if args.invert:
            passes = not passes
        if passes:
            kept.append({"path": path, "similarity": round(sim, 4)})
        else:
            removed.append({"path": path, "similarity": round(sim, 4)})

    # Sort kept by descending similarity so best matches are first
    kept.sort(key=lambda x: x["similarity"], reverse=True)

    elapsed = time.perf_counter() - t0
    print(f"\nQuery     : \"{args.query}\"")
    print(f"Threshold : {args.threshold}  {'(inverted)' if args.invert else ''}")
    print(f"Kept      : {len(kept)} / {len(video_paths)}  ({100*len(kept)/max(len(video_paths),1):.1f}%)")
    print(f"Removed   : {len(removed)}")
    print(f"Time      : {elapsed:.2f}s")

    if kept:
        print(f"\nTop 5 matches:")
        for r in kept[:5]:
            print(f"  {r['similarity']:.4f}  {Path(r['path']).name}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "query":     args.query,
            "threshold": args.threshold,
            "inverted":  args.invert,
            "total":     len(video_paths),
            "kept":      kept,
            "removed":   [r["path"] for r in removed],
        }, f, indent=2)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
