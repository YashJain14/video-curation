"""
prefetch_models.py
------------------
Run ONCE on the login node before qsub. Pre-populates the HuggingFace
cache with every model the pipeline loads.

Compute nodes on the cluster have no outbound internet (HF_HUB_OFFLINE=1).
If a model is not in the cache, the worker dies with
`huggingface_hub.errors.OfflineModeIsEnabled`. This script avoids that by
fetching everything up front from a node that does have internet.

Usage:
  conda activate video_curation_env
  python prefetch_models.py
"""

from huggingface_hub import hf_hub_download, snapshot_download

# Mirror the model IDs declared in embed.py / score.py / caption.py / filter.py.
CLIP_BASE   = "openai/clip-vit-base-patch32"          # embed.py, filter.py
CLIP_LARGE  = "openai/clip-vit-large-patch14"         # score.py
AESTHETIC_REPO  = "camenduru/improved-aesthetic-predictor"
AESTHETIC_FILE  = "sac+logos+ava1-l14-linearMSE.pth"
QWEN_VL     = "Qwen/Qwen2.5-VL-7B-Instruct"           # caption.py


def main():
    print(f"Prefetching {CLIP_BASE} ...")
    snapshot_download(repo_id=CLIP_BASE)

    print(f"Prefetching {CLIP_LARGE} ...")
    snapshot_download(repo_id=CLIP_LARGE)

    print(f"Prefetching {AESTHETIC_REPO}/{AESTHETIC_FILE} ...")
    hf_hub_download(repo_id=AESTHETIC_REPO, filename=AESTHETIC_FILE)

    print(f"Prefetching {QWEN_VL} (~16 GB, this is the slow one) ...")
    snapshot_download(repo_id=QWEN_VL)

    print("\nDone. All models are in the HuggingFace cache.")
    print("You can now submit run_curation.pbs from a compute node with HF_HUB_OFFLINE=1.")


if __name__ == "__main__":
    main()
