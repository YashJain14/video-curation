"""
prefetch_models.py
------------------
Run ONCE on the login node before qsub. Pre-populates the HuggingFace
cache with every model the pipeline loads.

Compute nodes on the cluster have no outbound internet (HF_HUB_OFFLINE=1
is set by the PBS script). If a model is not in the cache, the worker
dies with `huggingface_hub.errors.OfflineModeIsEnabled`. This script
avoids that by fetching everything up front from a node that does have
internet.

If your cluster sets HF_HUB_OFFLINE=1 globally (in /etc/profile or your
.bashrc), this script forcibly clears it for its own process so the
prefetch can actually reach HuggingFace.

Usage:
  conda activate video_curation_env
  python prefetch_models.py
"""

import os
import sys

# Strip any cluster-default offline flags BEFORE importing huggingface_hub.
# This file is the one place where we explicitly want network access.
for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    if os.environ.pop(var, None):
        print(f"  (unset {var} for this process)")

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub import constants as hf_constants

# Mirror the model IDs declared in embed.py / score.py / caption.py / filter.py.
CLIP_BASE   = "openai/clip-vit-base-patch32"          # embed.py, filter.py
CLIP_LARGE  = "openai/clip-vit-large-patch14"         # score.py
AESTHETIC_REPO  = "shunk031/improved-aesthetic-predictor"
AESTHETIC_FILE  = "sac+logos+ava1-l14-linearMSE.pth"
QWEN_VL     = "Qwen/Qwen2.5-VL-7B-Instruct"           # caption.py


def _fetch_repo(repo_id: str):
    print(f"\n→ {repo_id}")
    path = snapshot_download(repo_id=repo_id)
    print(f"  cached at: {path}")


def _fetch_file(repo_id: str, filename: str):
    print(f"\n→ {repo_id}/{filename}")
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    print(f"  cached at: {path}")


def main():
    print(f"HF cache dir: {hf_constants.HF_HUB_CACHE}")
    print("(set HF_HOME or HF_HUB_CACHE to override)")

    _fetch_repo(CLIP_BASE)
    _fetch_repo(CLIP_LARGE)
    _fetch_file(AESTHETIC_REPO, AESTHETIC_FILE)
    print("\n→ Qwen2.5-VL-7B (~16 GB, this one is slow) ...")
    _fetch_repo(QWEN_VL)

    print("\nAll models cached. You can now submit run_curation.pbs.")
    print(f"Cache: {hf_constants.HF_HUB_CACHE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: prefetch failed: {e}", file=sys.stderr)
        print("If this is OfflineModeIsEnabled, you ran prefetch on a node "
              "with no internet — run it on the LOGIN node, not a compute node.",
              file=sys.stderr)
        sys.exit(1)
