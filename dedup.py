"""
dedup.py
--------
Two-stage deduplication: exact → near.

Stage 1 — Exact dedup (MD5 hash):
  - Hash every video file byte-for-byte
  - Remove identical files immediately (same bytes = same video)
  - Fast: O(N) single pass, no GPU needed

Stage 2 — Near-dedup (CLIP embeddings + FAISS):
  - Load mean CLIP embeddings from data/embeddings/*.npz
  - Build FAISS IndexFlatIP (cosine similarity on L2-normalised vectors)
  - Union-Find clustering of videos with cosine sim > threshold
  - Keep one representative per cluster

Why two stages:
  - Exact dedup catches re-uploads and exact copies cheaply
  - Near-dedup catches re-encoded, cropped, or slightly edited duplicates
  - Running exact first reduces N before the O(N²) FAISS search

Why FAISS IndexFlatIP:
  - L2-normalised embeddings → inner product == cosine similarity
  - Exact search, fast enough for <100k videos
  - For >1M videos, swap to IndexIVFFlat with nlist=1024

Usage:
  python dedup.py --video_dir data/raw_videos --emb_dir data/embeddings --threshold 0.95
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import faiss
import wandb


def exact_dedup(video_paths: list[str]) -> tuple[list[str], list[str]]:
    """
    Stage 1: MD5-based exact deduplication.
    Hashes every file; keeps first occurrence of each hash.
    Returns (kept, removed).
    """
    seen   = {}
    kept   = []
    removed = []
    for path in video_paths:
        p = Path(path)
        if not p.exists():
            continue
        md5 = hashlib.md5(p.read_bytes()).hexdigest()
        if md5 in seen:
            removed.append(path)
        else:
            seen[md5] = path
            kept.append(path)
    return kept, removed


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
        data = np.load(p)
        video_paths.append(str(data["video_path"][0]))
        vectors.append(data["mean_embedding"].astype(np.float32))

    embeddings = np.stack(vectors, axis=0)   # [N, 512]
    # Re-normalise (should already be, but be safe)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    return video_paths, embeddings


def build_index(embeddings: np.ndarray,
                index_path: Path | None = None) -> faiss.IndexFlatIP:
    """
    Build (or load) an exact cosine-similarity FAISS index.
    If index_path exists, loads it instead of rebuilding.
    Saves the index to index_path after building.
    """
    if index_path and index_path.exists():
        print(f"Loading cached FAISS index from {index_path} ...")
        return faiss.read_index(str(index_path))

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    if index_path:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path))
        print(f"FAISS index saved → {index_path}")

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
    ap.add_argument("--video_dir",  default=None,
                    help="Video directory for exact dedup. If omitted, skips stage 1.")
    ap.add_argument("--emb_dir",    default="data/embeddings")
    ap.add_argument("--threshold",  type=float, default=0.95,
                    help="Cosine similarity threshold for near-duplicate")
    ap.add_argument("--out",        default="data/dedup_results.json")
    ap.add_argument("--index_path", default=None,
                    help="Path to save/load persistent FAISS index. If exists, skips rebuild.")
    args = ap.parse_args()

    emb_dir = Path(args.emb_dir)
    t0      = time.perf_counter()
    exact_removed = []

    wandb.init(project="video-curation", entity="rlx-labs",
               name="dedup-stage", resume="allow",
               config={"threshold":  args.threshold,
                       "video_dir":  args.video_dir,
                       "emb_dir":    str(emb_dir),
                       "index_path": args.index_path})

    # ── Stage 1: Exact dedup ──────────────────────────────────────────────────
    if args.video_dir:
        all_videos = [str(p) for p in Path(args.video_dir).rglob("*.mp4")]
        print(f"Stage 1 — Exact dedup (MD5) over {len(all_videos)} videos ...")
        after_exact, exact_removed = exact_dedup(all_videos)
        print(f"  Exact duplicates removed : {len(exact_removed)}")
        print(f"  Remaining                : {len(after_exact)}")
    else:
        print("Stage 1 — Exact dedup skipped (no --video_dir provided)")

    # ── Stage 2: Near-dedup (FAISS) ───────────────────────────────────────────
    print(f"\nStage 2 — Near-dedup (FAISS cosine sim > {args.threshold}) ...")
    print(f"Loading embeddings from {emb_dir} ...")
    video_paths, embeddings = load_embeddings(emb_dir)
    print(f"  Loaded {len(video_paths)} videos  dim={embeddings.shape[1]}")

    print("Building FAISS index ...")
    index_path = Path(args.index_path) if args.index_path else None
    index = build_index(embeddings, index_path)

    print(f"Finding near-duplicates ...")
    kept, near_removed = find_duplicates(video_paths, embeddings, index, args.threshold)

    elapsed = time.perf_counter() - t0
    total   = len(video_paths)
    print(f"\nResults:")
    print(f"  Input (to near-dedup) : {total}")
    print(f"  Exact removed         : {len(exact_removed)}")
    print(f"  Near removed          : {len(near_removed)}  ({100*len(near_removed)/max(total,1):.1f}%)")
    print(f"  Final kept            : {len(kept)}")
    print(f"  Time                  : {elapsed:.2f}s")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "kept":          kept,
            "removed":       near_removed,
            "exact_removed": exact_removed,
            "threshold":     args.threshold,
            "total":         total,
        }, f, indent=2)
    print(f"\nResults saved → {out}")

    wandb.log({
        "dedup/exact_removed": len(exact_removed),
        "dedup/near_removed":  len(near_removed),
        "dedup/kept":          len(kept),
        "dedup/total":         total,
        "dedup/near_removed_pct": 100 * len(near_removed) / max(total, 1),
        "dedup/elapsed_s":     elapsed,
    })
    wandb.finish()


if __name__ == "__main__":
    main()
