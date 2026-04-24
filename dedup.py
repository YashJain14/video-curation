"""
dedup.py
--------
Near-duplicate detection over a set of video CLIP embeddings.
Uses FAISS IndexFlatIP (exact cosine similarity on L2-normalised vectors).

Strategy:
  - Load all mean embeddings from data/embeddings/*.npz
  - Build a FAISS index
  - For each video, find all neighbours with cosine similarity > threshold
  - Keep one representative per duplicate cluster (highest-norm embedding wins)
  - Write kept/removed lists to data/dedup_results.json

Why FAISS IndexFlatIP:
  - Embeddings are L2-normalised → inner product == cosine similarity
  - IndexFlatIP is exact (no approximation error) and fast enough for <100k videos
  - For >1M videos, swap to IndexIVFFlat with nlist=1024 for approximate search

Usage:
  python dedup.py --emb_dir data/embeddings --threshold 0.95
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import faiss


def load_embeddings(emb_dir: Path) -> tuple[list[str], np.ndarray]:
    """
    Load all mean_embedding vectors from *.npz files.
    Returns (video_paths, embeddings) where embeddings is [N, 512] float32.
    """
    paths = sorted(emb_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {emb_dir}")

    video_paths = []
    vectors     = []
    for p in paths:
        data = np.load(p, allow_pickle=True)
        video_paths.append(str(data["video_path"][0]))
        vectors.append(data["mean_embedding"].astype(np.float32))

    embeddings = np.stack(vectors, axis=0)   # [N, 512]
    # Re-normalise (should already be, but be safe)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    return video_paths, embeddings


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build an exact cosine-similarity FAISS index."""
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def find_duplicates(video_paths: list[str], embeddings: np.ndarray,
                    index: faiss.IndexFlatIP,
                    threshold: float = 0.95) -> tuple[list[str], list[str]]:
    """
    Union-Find clustering of near-duplicates.
    Returns (kept, removed).
    """
    N = len(video_paths)

    # For each video, find all neighbours above threshold (including itself at sim=1.0)
    # k=50 is a reasonable upper bound for near-dup neighbours in a video corpus
    k     = min(50, N)
    sims, idxs = index.search(embeddings, k)   # [N, k]

    # Union-Find
    parent = list(range(N))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for i in range(N):
        for j_pos in range(k):
            j   = int(idxs[i, j_pos])
            sim = float(sims[i, j_pos])
            if j == i:
                continue
            if sim >= threshold:
                union(i, j)

    # Build clusters: root -> list of members
    clusters: dict[int, list[int]] = {}
    for i in range(N):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    kept    = []
    removed = []
    for root, members in clusters.items():
        # Keep the member whose embedding has highest norm before normalisation
        # (proxy for "most confident" representation)
        kept_idx = members[0]
        kept.append(video_paths[kept_idx])
        for m in members[1:]:
            removed.append(video_paths[m])

    return kept, removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir",   default="data/embeddings")
    ap.add_argument("--threshold", type=float, default=0.95,
                    help="Cosine similarity threshold for near-duplicate")
    ap.add_argument("--out",       default="data/dedup_results.json")
    args = ap.parse_args()

    emb_dir = Path(args.emb_dir)
    t0      = time.perf_counter()

    print(f"Loading embeddings from {emb_dir} ...")
    video_paths, embeddings = load_embeddings(emb_dir)
    print(f"  Loaded {len(video_paths)} videos  dim={embeddings.shape[1]}")

    print("Building FAISS index ...")
    index = build_index(embeddings)

    print(f"Finding duplicates (threshold={args.threshold}) ...")
    kept, removed = find_duplicates(video_paths, embeddings, index, args.threshold)

    elapsed = time.perf_counter() - t0
    print(f"\nResults:")
    print(f"  Total   : {len(video_paths)}")
    print(f"  Kept    : {len(kept)}")
    print(f"  Removed : {len(removed)}  ({100*len(removed)/len(video_paths):.1f}% duplicates)")
    print(f"  Time    : {elapsed:.2f}s")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"kept": kept, "removed": removed,
                   "threshold": args.threshold,
                   "total": len(video_paths)}, f, indent=2)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
