"""
manifest.py
-----------
Dataset versioning via a JSON manifest.

Each pipeline run produces a versioned manifest that records:
  - Version string (v1, v2, ...)
  - Timestamp
  - Pipeline config (thresholds, model IDs, frame counts)
  - Input: list of source videos
  - Output: list of shard paths + per-shard stats
  - Dedup stats, score stats

This allows any training run to be reproduced exactly by re-running
the pipeline with the same manifest config.

Usage:
  # Create a new manifest
  python manifest.py create \
    --version v1 \
    --video_dir data/raw_videos \
    --dedup data/dedup_results.json \
    --scores data/scores.json \
    --shards data/shards \
    --out data/manifests/v1.json

  # List all manifests
  python manifest.py list --manifest_dir data/manifests

  # Compare two versions
  python manifest.py diff --a data/manifests/v1.json --b data/manifests/v2.json
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path


def create_manifest(version: str, video_dir: str, dedup_path: str,
                    scores_path: str, shards_dir: str,
                    config: dict, out_path: str):

    # Source videos
    all_videos = sorted(str(p) for p in Path(video_dir).rglob("*.mp4"))

    # Dedup stats
    dedup_stats = {}
    if Path(dedup_path).exists():
        with open(dedup_path) as f:
            dedup = json.load(f)
        dedup_stats = {
            "total":   dedup.get("total", len(all_videos)),
            "kept":    len(dedup.get("kept", [])),
            "removed": len(dedup.get("removed", [])),
            "threshold": dedup.get("threshold", None),
        }

    # Score stats
    score_stats = {}
    if Path(scores_path).exists():
        with open(scores_path) as f:
            scores = json.load(f)
        valid   = [s["mean_score"] for s in scores if s.get("status") == "ok"]
        passing = [s for s in valid if s >= config.get("min_score", 4.5)]
        score_stats = {
            "scored":  len(valid),
            "passing": len(passing),
            "mean":    round(sum(valid) / max(len(valid), 1), 3),
            "min_score_threshold": config.get("min_score", 4.5),
        }

    # Shard stats
    shard_files = sorted(Path(shards_dir).glob("*.tar")) if Path(shards_dir).exists() else []
    shards_info = []
    total_bytes = 0
    for s in shard_files:
        sz = s.stat().st_size
        total_bytes += sz
        shards_info.append({"name": s.name, "size_bytes": sz})

    manifest = {
        "version":    version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config":     config,
        "sources": {
            "video_dir":    video_dir,
            "total_videos": len(all_videos),
        },
        "dedup":   dedup_stats,
        "scoring": score_stats,
        "shards": {
            "count":       len(shard_files),
            "total_bytes": total_bytes,
            "total_mb":    round(total_bytes / 1e6, 1),
            "files":       shards_info,
        },
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved → {out}")
    print(f"  Videos  : {len(all_videos)}")
    print(f"  Dedup   : {dedup_stats.get('kept', '?')} kept")
    print(f"  Scoring : {score_stats.get('passing', '?')} passing")
    print(f"  Shards  : {len(shard_files)} files  ({round(total_bytes/1e6,1)} MB)")


def list_manifests(manifest_dir: str):
    manifests = sorted(Path(manifest_dir).glob("*.json"))
    if not manifests:
        print(f"No manifests found in {manifest_dir}")
        return
    print(f"{'Version':<12} {'Created':<28} {'Videos':>8} {'Kept':>8} {'Shards':>8} {'Size MB':>10}")
    print("-" * 80)
    for m in manifests:
        with open(m) as f:
            d = json.load(f)
        print(
            f"{d.get('version','?'):<12} "
            f"{d.get('created_at','?')[:19]:<28} "
            f"{d.get('sources',{}).get('total_videos','?'):>8} "
            f"{d.get('dedup',{}).get('kept','?'):>8} "
            f"{d.get('shards',{}).get('count','?'):>8} "
            f"{d.get('shards',{}).get('total_mb','?'):>10}"
        )


def diff_manifests(path_a: str, path_b: str):
    with open(path_a) as f: a = json.load(f)
    with open(path_b) as f: b = json.load(f)

    print(f"Diff: {a['version']} → {b['version']}")
    print()

    def show(label, va, vb):
        delta = ""
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            delta = f"  (Δ {vb - va:+.1f})"
        print(f"  {label:<30} {str(va):>12} → {str(vb):>12}{delta}")

    show("Total videos",   a["sources"]["total_videos"],       b["sources"]["total_videos"])
    show("Dedup kept",     a["dedup"].get("kept","?"),          b["dedup"].get("kept","?"))
    show("Dedup threshold",a["dedup"].get("threshold","?"),     b["dedup"].get("threshold","?"))
    show("Scoring passing",a["scoring"].get("passing","?"),     b["scoring"].get("passing","?"))
    show("Min score",      a["scoring"].get("min_score_threshold","?"), b["scoring"].get("min_score_threshold","?"))
    show("Shards",         a["shards"]["count"],                b["shards"]["count"])
    show("Total MB",       a["shards"]["total_mb"],             b["shards"]["total_mb"])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    # create
    cp = sub.add_parser("create")
    cp.add_argument("--version",   required=True)
    cp.add_argument("--video_dir", required=True)
    cp.add_argument("--dedup",     required=True)
    cp.add_argument("--scores",    required=True)
    cp.add_argument("--shards",    required=True)
    cp.add_argument("--out",       required=True)
    cp.add_argument("--min_score", type=float, default=4.5)
    cp.add_argument("--dedup_threshold", type=float, default=0.95)

    # list
    lp = sub.add_parser("list")
    lp.add_argument("--manifest_dir", default="data/manifests")

    # diff
    dp = sub.add_parser("diff")
    dp.add_argument("--a", required=True)
    dp.add_argument("--b", required=True)

    args = ap.parse_args()

    if args.cmd == "create":
        config = {
            "min_score":        args.min_score,
            "dedup_threshold":  args.dedup_threshold,
        }
        create_manifest(args.version, args.video_dir, args.dedup,
                        args.scores, args.shards, config, args.out)
    elif args.cmd == "list":
        list_manifests(args.manifest_dir)
    elif args.cmd == "diff":
        diff_manifests(args.a, args.b)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
