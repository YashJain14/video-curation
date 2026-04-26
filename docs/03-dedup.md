# Stage 3 — dedup.py · Deduplication

Two-stage deduplication to remove exact and near-duplicate videos. Outputs `dedup_results.json` with a `kept` list used by all downstream stages.

---

## What it does

### Stage 1 — MD5 Exact Dedup

Hashes every video file byte-for-byte with MD5. Keeps the first occurrence of each hash, discards subsequent identical files. O(N) single pass, no GPU needed.

```python
md5 = hashlib.md5(p.read_bytes()).hexdigest()
```

Catches re-uploads and truly identical files before the more expensive FAISS step.

### Stage 2 — FAISS Near-Dedup

1. Loads all `mean_embedding [512]` vectors from `embeddings/*.npz`.
2. Re-normalises to unit L2 (should already be from embed.py, but defensive).
3. Builds (or loads from disk) a `faiss.IndexFlatIP` — an exact inner-product index. Because vectors are L2-normalised, inner product equals cosine similarity.
4. Runs `index.search(embeddings, k=50)` to find up to 50 nearest neighbours per video.
5. Union-Find clusters all pairs with cosine similarity ≥ threshold.
6. Keeps `members[0]` of each cluster, discards the rest.

---

## Inputs

| Path                  | Description                              |
|-----------------------|------------------------------------------|
| `raw_videos/**/*.mp4` | Source videos (for exact dedup via MD5)  |
| `embeddings/*.npz`    | CLIP embeddings from Stage 2             |

---

## Outputs

| Path                   | Description                                                    |
|------------------------|----------------------------------------------------------------|
| `dedup_results.json`   | `{kept, removed, exact_removed, threshold, total}`             |
| `faiss.index`          | Persistent FAISS index (reused on reruns, avoids O(N²) rebuild)|

### dedup_results.json schema

```json
{
  "kept":          ["path/to/video.mp4", ...],
  "removed":       ["path/to/dup.mp4", ...],
  "exact_removed": ["path/to/exact_dup.mp4", ...],
  "threshold":     0.95,
  "total":         9998
}
```

The `kept` list is consumed by `shard.py` to filter the final dataset.

---

## CLI

```bash
python dedup.py \
  --video_dir  $SCRATCH_DIR/raw_videos \
  --emb_dir    $SCRATCH_DIR/embeddings \
  --threshold  0.95 \
  --out        $SCRATCH_DIR/dedup_results.json \
  --index_path $SCRATCH_DIR/faiss.index \
  --md5_cache  $SCRATCH_DIR/md5_cache.json
```

| Argument       | Default                      | Description                              |
|----------------|------------------------------|------------------------------------------|
| `--video_dir`  | `None`                       | Skip exact dedup if omitted              |
| `--emb_dir`    | `data/embeddings`            | Directory with .npz embedding files      |
| `--threshold`  | `0.95`                       | Cosine similarity cutoff for near-dedup  |
| `--out`        | `data/dedup_results.json`    | Output JSON path                         |
| `--index_path` | `None`                       | Path to save/load FAISS index            |
| `--md5_cache`  | `None`                       | JSON cache of {path: md5}; skips re-reading unchanged video files on reruns |

---

## Union-Find Algorithm

```python
parent = list(range(N))

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]  # path compression
        x = parent[x]
    return x

def union(a, b):
    parent[find(a)] = find(b)

# For each video, union with all neighbours above threshold
for i in range(N):
    for j_pos in range(k):
        j, sim = int(idxs[i, j_pos]), float(sims[i, j_pos])
        if j != i and sim >= threshold:
            union(i, j)
```

Path compression keeps `find()` near O(1). The cluster representative is always `members[0]` — the first video encountered for that root.

---

## FAISS Index Details

**Type:** `IndexFlatIP` (flat inner product)

Why not IVF or HNSW:
- Dataset size is <100k videos — exact search is fast enough (~91s for 10k)
- `IndexFlatIP` gives guaranteed exact results, no approximation error
- For >500k videos, upgrade to `IndexIVFFlat(nlist=1024)` for sub-linear query time

**Persistence:** The index is saved to `faiss.index` after build. On reruns the pre-built index is loaded directly, skipping the O(N²) rebuild.

---

## Run Metrics

| Metric | Value |
|--------|-------|
| Input videos | 9,998 |
| Exact duplicates removed | 0 |
| Near-duplicates removed | 54 (0.54%) |
| Final kept | 9,944 |
| Elapsed (cold, no md5_cache) | 10:15 |
| Elapsed (warm, md5_cache hit) | ~2 min (skips ~40 GB of video I/O) |

**MD5 cache:** On reruns, `md5_cache.json` stores `{path: md5}` so unmodified files are not re-read from disk. With 10k videos averaging ~4 MB each, cold MD5 hashing reads ~40 GB; the cache reduces this to near-zero on subsequent runs.

---

## wandb Metrics Logged

| Metric                    | Description                        |
|---------------------------|------------------------------------|
| `dedup/exact_removed`     | MD5 exact duplicates removed       |
| `dedup/near_removed`      | FAISS near-duplicates removed      |
| `dedup/kept`              | Videos remaining after both stages |
| `dedup/total`             | Input to near-dedup stage          |
| `dedup/near_removed_pct`  | Near-duplicate percentage          |
| `dedup/elapsed_s`         | Wall time                          |

---

## Why Two Stages

| Stage         | Complexity | Catches                                      |
|---------------|-----------|----------------------------------------------|
| MD5 exact     | O(N)      | Bit-identical re-uploads                     |
| FAISS near    | O(N²)     | Re-encoded, cropped, slightly edited copies  |

Running exact first reduces N before the expensive FAISS search. In practice Kinetics-400 has zero exact duplicates, but in real production pipelines from the web exact copies are common.
