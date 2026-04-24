#!/bin/bash
# ---------------------------------------------------------------------------
# Setup Script: Run on the LOGIN NODE before submitting run_curation.pbs
# ---------------------------------------------------------------------------

echo "Setting up video_curation_env ..."

CONDA_DIR="$HOME/miniconda3"
source "$CONDA_DIR/etc/profile.d/conda.sh"

ENV_NAME="video_curation_env"
conda create -y -n $ENV_NAME python=3.10
conda activate $ENV_NAME

# Core
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install numpy matplotlib

# Video decode
pip install pynvvideocodec
pip install opencv-python-headless

# CLIP + embeddings
pip install transformers accelerate
pip install faiss-gpu

# Aesthetic scorer
pip install huggingface_hub

# VLM captioning (Qwen2.5-VL)
pip install qwen-vl-utils
# transformers already installed above

# WebDataset
pip install webdataset

# Orchestration
pip install prefect

# Kinetics downloader
pip install yt-dlp

# HTTP downloads (ingest)
pip install requests

# Ray distributed
pip install "ray[default]"

# Profiling: torchscope (local checkout, not on PyPI)
# Clone your torchscope repo and install as editable:
#   git clone https://github.com/YashJain14/torchscope.git ~/torchscope
#   pip install -e ~/torchscope
# Then pass --torchscope ~/torchscope when running profile_run.py

echo ""
echo "Setup complete. Submit with: qsub run_curation.pbs"
