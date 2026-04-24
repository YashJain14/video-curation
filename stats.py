"""
stats.py
--------
Dataset composition analysis.

Loads scores, captions, embeddings, and dedup results to produce:
  1. Label distribution — class balance across Kinetics action categories
  2. Score histogram    — aesthetic quality distribution
  3. Dedup summary      — how much near-duplication was found
  4. Caption coverage   — how many clips have captions
  5. CLIP embedding PCA — 2D projection of semantic space (saved as PNG)
  6. Underrepresented classes — labels with fewer clips than threshold

All stats printed to stdout and saved to data/dataset_stats.json.
Chart saved to data/stats_embedding_pca.png.

Usage:
  python stats.py --video_dir data/raw_videos --scores data/scores.json \
                  --captions data/captions.json --dedup data/dedup_results.json \
                  --emb_dir data/embeddings
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def label_distribution(video_dir: Path) -> dict:
    """Count clips per Kinetics action label (subdirectory name)."""
    counts = Counter()
    for mp4 in video_dir.rglob("*.mp4"):
        counts[mp4.parent.name] += 1
    return dict(counts.most_common())


def score_stats(scores_path: Path) -> dict:
    with open(scores_path) as f:
        records = json.load(f)
    valid = [r["mean_score"] for r in records if r.get("status") == "ok"]
    if not valid:
        return {}
    arr = np.array(valid)
    return {
        "n":        len(arr),
        "mean":     round(float(arr.mean()), 3),
        "median":   round(float(np.median(arr)), 3),
        "std":      round(float(arr.std()), 3),
        "min":      round(float(arr.min()), 3),
        "max":      round(float(arr.max()), 3),
        "pct_above_4.5": round(float((arr >= 4.5).mean() * 100), 1),
        "pct_above_5.0": round(float((arr >= 5.0).mean() * 100), 1),
    }


def dedup_stats(dedup_path: Path) -> dict:
    with open(dedup_path) as f:
        d = json.load(f)
    total   = d.get("total", 0)
    kept    = len(d.get("kept", []))
    removed = len(d.get("removed", []))
    return {
        "total":           total,
        "kept":            kept,
        "removed":         removed,
        "duplicate_pct":   round(100 * removed / max(total, 1), 1),
        "threshold":       d.get("threshold"),
    }


def caption_coverage(captions_path: Path, video_dir: Path) -> dict:
    with open(captions_path) as f:
        caps = json.load(f)
    total_videos  = len(list(video_dir.rglob("*.mp4")))
    captioned     = sum(1 for c in caps if c.get("status") == "ok" and c.get("caption"))
    empty_captions = sum(1 for c in caps if c.get("status") == "ok" and not c.get("caption"))
    return {
        "total_videos":  total_videos,
        "captioned":     captioned,
        "coverage_pct":  round(100 * captioned / max(total_videos, 1), 1),
        "empty":         empty_captions,
        "failed":        sum(1 for c in caps if c.get("status") != "ok"),
    }


def underrepresented_classes(label_counts: dict, min_clips: int = 10) -> list[str]:
    return [label for label, count in label_counts.items() if count < min_clips]


def plot_score_histogram(scores_path: Path, out_path: Path):
    with open(scores_path) as f:
        records = json.load(f)
    scores = [r["mean_score"] for r in records if r.get("status") == "ok"]
    if not scores:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores, bins=40, color="#4A90D9", edgecolor="white", linewidth=0.5)
    ax.axvline(4.5, color="#E67E22", linestyle="--", linewidth=1.5, label="threshold=4.5")
    ax.axvline(5.0, color="#27AE60", linestyle="--", linewidth=1.5, label="threshold=5.0")
    ax.set_xlabel("Aesthetic Score")
    ax.set_ylabel("Count")
    ax.set_title("Aesthetic Score Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Score histogram → {out_path}")


def plot_label_distribution(label_counts: dict, out_path: Path, top_n: int = 30):
    if not label_counts:
        return
    labels = list(label_counts.keys())[:top_n]
    counts = [label_counts[l] for l in labels]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(labels)), counts, color="#7B68EE", edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace("_", " ") for l in labels],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Clip count")
    ax.set_title(f"Label Distribution (top {top_n} classes)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Label distribution → {out_path}")


def plot_embedding_pca(emb_dir: Path, out_path: Path, max_points: int = 2000):
    """PCA projection of CLIP embeddings coloured by label."""
    from sklearn.decomposition import PCA

    npz_files = sorted(emb_dir.glob("*.npz"))[:max_points]
    if not npz_files:
        return

    vectors, labels = [], []
    for p in npz_files:
        data  = np.load(p, allow_pickle=True)
        vectors.append(data["mean_embedding"].astype(np.float32))
        video_path = str(data["video_path"][0])
        labels.append(Path(video_path).parent.name)

    X          = np.stack(vectors)
    pca        = PCA(n_components=2)
    X2         = pca.fit_transform(X)

    unique_labels = sorted(set(labels))
    cmap          = plt.cm.get_cmap("tab20", len(unique_labels))
    label_to_idx  = {l: i for i, l in enumerate(unique_labels)}
    colors        = [cmap(label_to_idx[l]) for l in labels]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(X2[:, 0], X2[:, 1], c=colors, s=8, alpha=0.6)
    ax.set_title(f"CLIP Embedding Space — PCA (n={len(vectors)})")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Embedding PCA → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--scores",    default="data/scores.json")
    ap.add_argument("--captions",  default="data/captions.json")
    ap.add_argument("--dedup",     default="data/dedup_results.json")
    ap.add_argument("--emb_dir",   default="data/embeddings")
    ap.add_argument("--out_dir",   default="data")
    ap.add_argument("--min_clips", type=int, default=10,
                    help="Flag classes with fewer clips than this")
    args = ap.parse_args()

    out_dir   = Path(args.out_dir)
    video_dir = Path(args.video_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {}

    print("Label distribution ...")
    label_counts = label_distribution(video_dir)
    stats["label_distribution"] = label_counts
    stats["n_classes"]          = len(label_counts)
    stats["total_clips"]        = sum(label_counts.values())
    under = underrepresented_classes(label_counts, args.min_clips)
    stats["underrepresented_classes"] = under
    print(f"  {len(label_counts)} classes  |  {stats['total_clips']} clips  |  "
          f"{len(under)} classes with <{args.min_clips} clips")

    if Path(args.scores).exists():
        print("Score stats ...")
        stats["scores"] = score_stats(Path(args.scores))
        s = stats["scores"]
        print(f"  mean={s.get('mean')}  median={s.get('median')}  "
              f"≥4.5: {s.get('pct_above_4.5')}%  ≥5.0: {s.get('pct_above_5.0')}%")

    if Path(args.dedup).exists():
        print("Dedup stats ...")
        stats["dedup"] = dedup_stats(Path(args.dedup))
        d = stats["dedup"]
        print(f"  {d['removed']} duplicates removed ({d['duplicate_pct']}%)")

    if Path(args.captions).exists():
        print("Caption coverage ...")
        stats["captions"] = caption_coverage(Path(args.captions), video_dir)
        c = stats["captions"]
        print(f"  {c['captioned']}/{c['total_videos']} captioned ({c['coverage_pct']}%)")

    # Charts
    print("\nGenerating charts ...")
    if Path(args.scores).exists():
        plot_score_histogram(Path(args.scores), out_dir / "stats_score_histogram.png")
    plot_label_distribution(label_counts, out_dir / "stats_label_distribution.png")
    if Path(args.emb_dir).exists():
        try:
            plot_embedding_pca(Path(args.emb_dir), out_dir / "stats_embedding_pca.png")
        except ImportError:
            print("  Skipping PCA plot (pip install scikit-learn)")

    # Save JSON
    out_json = out_dir / "dataset_stats.json"
    with open(out_json, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved → {out_json}")

    # Print summary
    print(f"\n{'─'*50}")
    print(f"  Dataset Summary")
    print(f"{'─'*50}")
    print(f"  Classes   : {stats.get('n_classes', '?')}")
    print(f"  Clips     : {stats.get('total_clips', '?')}")
    if "dedup" in stats:
        print(f"  After dedup: {stats['dedup']['kept']}")
    if "scores" in stats:
        print(f"  Passing score (≥4.5): {stats['scores']['pct_above_4.5']}%")
    if "captions" in stats:
        print(f"  Caption coverage: {stats['captions']['coverage_pct']}%")
    if under:
        print(f"  Underrepresented (<{args.min_clips} clips): {under[:5]}{'...' if len(under)>5 else ''}")
    print(f"{'─'*50}")


if __name__ == "__main__":
    main()
