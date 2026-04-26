# Setup & Cluster Configuration

Everything needed to set up the environment, prefetch model weights, and submit jobs on the NSCC HPC cluster.

---

## One-Time Setup on the Login Node

```bash
# 1. Create conda env and install all dependencies
bash setup_env.sh

# 2. Pre-fetch all model weights (compute nodes have no internet)
python prefetch_models.py

# 3. Clone torchscope (for GPU profiling)
git clone https://github.com/YashJain14/torchscope.git ~/torchscope
```

---

## setup_env.sh

Creates a `video_curation_env` conda environment with Python 3.10 and installs all dependencies.

### Dependency Notes

| Package                          | Pin / Note                                                                    |
|----------------------------------|-------------------------------------------------------------------------------|
| `numpy<2`                        | faiss-gpu (legacy) is compiled against NumPy 1.x ABI — pin prevents crash    |
| `opencv-python-headless<4.11`    | 4.11+ built against NumPy 2 ABI, conflicts with numpy<2 pin                   |
| `torch` / `torchvision`          | Installed from `download.pytorch.org/whl/cu124` for CUDA 12.4                |
| `qwen-vl-utils[decord]`          | The decord extra is required for native video input in caption.py             |
| `nvidia-ml-py`                   | Installs as `pynvml` import — required for GPU util metrics in torchscope     |
| `faiss-gpu`                      | GPU-accelerated FAISS; CPU-only `faiss-cpu` works but is slower               |

### Environment Variables

`setup_env.sh` exports:
```bash
export WANDB_API_KEY=<your_wandb_api_key>
```

Replace with your actual key from https://wandb.ai/authorize.

---

## prefetch_models.py

Pre-downloads all model weights into the HuggingFace cache **on the login node**. Compute nodes run with `HF_HUB_OFFLINE=1` and will fail if any model is missing.

### Models fetched

| Model                                  | Used in          | Size    |
|----------------------------------------|------------------|---------|
| `openai/clip-vit-base-patch32`         | embed.py, filter.py | ~600 MB |
| `openai/clip-vit-large-patch14`        | score.py         | ~900 MB |
| `camenduru/improved-aesthetic-predictor` (`sac+logos+ava1-l14-linearMSE.pth`) | score.py | ~10 MB |
| `Qwen/Qwen3-VL-8B-Instruct`           | caption.py       | ~16 GB  |

**Total:** ~18 GB. The Qwen3-VL download is the slow step (~30 min on a fast connection).

```bash
python prefetch_models.py
# Done. All models are in the HuggingFace cache.
```

---

## PBS Job Files

### run_curation.pbs — Full Pipeline

```
#PBS -N video_curation
#PBS -l select=1:ngpus=4
#PBS -l walltime=02:00:00
#PBS -q ai
```

Key environment setup:
```bash
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export SCRATCH_DIR="$HOME/scratch/video-curation"
```

Runs `python dag.py` with production parameters. Check status:
```bash
qstat -u $USER
tail -f curation_output.log
```

### run_profile.pbs — GPU Profiling

```
#PBS -l select=1:ngpus=4
#PBS -l walltime=00:30:00
```

Runs `profile_run.py` with `--force --limit 500`. Download reports after:
```bash
scp user@cluster:~/scratch/video-curation/reports/*.html .
```

### run_diag.pbs — Diagnostics

Runs environment diagnostics (GPU visibility, library versions, torch CUDA check). Useful for debugging environment issues.

---

## Full Requirements

`requirements.txt` contains the complete frozen environment. Key packages:

| Package            | Version   |
|--------------------|-----------|
| torch              | 2.6.0+cu124 |
| torchvision        | 0.21.0+cu124 |
| transformers       | 5.6.2     |
| ray                | 2.55.1    |
| prefect            | 3.6.27    |
| faiss-gpu          | 1.7.2     |
| numpy              | 1.26.4    |
| wandb              | 0.26.1    |
| qwen-vl-utils      | 0.0.14    |
| av                 | 17.0.1    |
| pynvvideocodec     | 2.1.0     |
| webdataset         | 1.0.2     |
| huggingface_hub    | 1.12.0    |
| decord             | 0.6.0     |
| matplotlib         | 3.10.9    |
| scikit-image       | 0.25.2    |

---

## Offline Mode

The cluster's compute nodes have no outbound internet. All HuggingFace model downloads must be done on the login node. The PBS scripts set:

```bash
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

If a model is missing from the cache, the worker will fail immediately with:
```
huggingface_hub.errors.OfflineModeIsEnabled
```

Run `prefetch_models.py` on the login node to prevent this.
