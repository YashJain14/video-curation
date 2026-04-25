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
# numpy pinned to <2: faiss-gpu (legacy) is compiled against NumPy 1.x ABI.
# Without the pin, `import faiss` crashes with `numpy.core.multiarray failed to import`.
pip install "numpy<2" matplotlib

# Video decode
pip install pynvvideocodec
# opencv-python-headless 4.11+ is built against NumPy 2 ABI; pin <4.11 to stay
# compatible with the numpy<2 pin above (faiss-gpu requires NumPy 1).
pip install "opencv-python-headless<4.11"

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

# Experiment tracking
pip install wandb
export WANDB_API_KEY=<your_wandb_api_key>

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
echo "Setup complete."
echo "Next: pre-fetch model weights from the login node (compute nodes are offline):"
echo "  python prefetch_models.py"
echo "Then submit:"
echo "  qsub run_curation.pbs"
