#!/usr/bin/env bash
set -euo pipefail

# Conda setup for the SEG.A aorta inference notebook.
#
# Usage:
#   bash setup_conda_environment.sh                    # default env name
#   ENV_NAME=seg-aorta bash setup_conda_environment.sh # custom environment name
#
# Why this script does not put every package in an environment.yml:
# A full Conda solve with PyTorch + MONAI + SimpleITK + pymeshfix can be very slow.
# This script creates a small Conda environment first, then installs the whole
# scientific/medical-imaging stack from requirements.txt with pip, which is much
# faster and more reliable.

ENV_NAME="${ENV_NAME:-sega-aorta}"
KERNEL_NAME="${KERNEL_NAME:-${ENV_NAME}}"
KERNEL_DISPLAY_NAME="${KERNEL_DISPLAY_NAME:-Python (${ENV_NAME})}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Install Miniconda, Miniforge, or Mambaforge first, then rerun this script."
  exit 1
fi

if [[ ! -f requirements.txt ]]; then
  echo "requirements.txt was not found. Run this script from the project root."
  exit 1
fi

# Load conda shell function for non-interactive shells.
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda environment '${ENV_NAME}' already exists. Reusing it."
else
  echo "Creating small Conda environment '${ENV_NAME}'..."
  conda create -y \
    --name "${ENV_NAME}" \
    --channel conda-forge \
    "python=${PYTHON_VERSION}" \
    "pip>=23" \
    "jupyterlab>=4,<5" \
    "ipywidgets>=8.1,<9" \
    wheel \
    setuptools
fi

conda activate "${ENV_NAME}"

echo "Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

echo "Installing all dependencies (including CUDA 12.1 torch) from requirements.txt..."
python -m pip install -r requirements.txt

echo "Installing Jupyter kernel '${KERNEL_NAME}'..."
python -m ipykernel install \
  --user \
  --name "${KERNEL_NAME}" \
  --display-name "${KERNEL_DISPLAY_NAME}"

echo "Running import check..."
python - <<'PY'
import torch
import monai
import SimpleITK
import nibabel
import skimage
import trimesh
import matplotlib
import imageio
import plotly
print("torch:", torch.__version__)
print("monai:", monai.__version__)
print("cuda available:", torch.cuda.is_available())
print("imageio:", imageio.__version__)
print("plotly:", plotly.__version__)
PY

cat <<MSG

Conda environment is ready: ${ENV_NAME}
Notebook kernel: ${KERNEL_DISPLAY_NAME}

Activate it with:
  conda activate ${ENV_NAME}

Start Jupyter with:
  jupyter lab

Expected project layout before running the notebook:
  models/checkpoint_segresnet.pth
  data/volumes_full/D1.nii.gz
  data/labels_full/D1.nii.gz
MSG
