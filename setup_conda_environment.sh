#!/usr/bin/env bash
set -euo pipefail

# Fast Conda setup for the SEG.A aorta inference notebook.
#
# Usage:
#   bash setup_conda_environment.sh                         # CPU PyTorch
#   TORCH_TARGET=cu118 bash setup_conda_environment.sh      # CUDA 11.8 PyTorch
#   ENV_NAME=seg-aorta bash setup_conda_environment.sh      # Custom environment name
#
# Why this script does not put every package in environment.yml:
# A full Conda solve with PyTorch + MONAI + SimpleITK + pymeshfix can be very slow.
# This script creates a small Conda environment first, then installs the scientific
# and medical-imaging stack with pip, which is usually much faster and more reliable.

TORCH_TARGET="${TORCH_TARGET:-cpu}"   # cpu | cu118
ENV_NAME="${ENV_NAME:-sega-aorta}"
KERNEL_NAME="${KERNEL_NAME:-${ENV_NAME}}"
KERNEL_DISPLAY_NAME="${KERNEL_DISPLAY_NAME:-Python (${ENV_NAME})}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Install Miniconda, Miniforge, or Mambaforge first, then rerun this script."
  exit 1
fi

if [[ ! -f requirements.txt ]]; then
  echo "requirements.txt was not found. Run this script from the tutorial package folder."
  exit 1
fi

case "${TORCH_TARGET}" in
  cpu)
    TORCH_REQUIREMENTS="requirements-torch-cpu.txt"
    ;;
  cu118)
    TORCH_REQUIREMENTS="requirements-torch-cu118.txt"
    ;;
  *)
    echo "Unknown TORCH_TARGET='${TORCH_TARGET}'. Use cpu or cu118."
    exit 1
    ;;
esac

if [[ ! -f "${TORCH_REQUIREMENTS}" ]]; then
  echo "Torch requirements file not found: ${TORCH_REQUIREMENTS}"
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
    "ipykernel>=6.29,<7" \
    "ipywidgets>=8.1,<9" \
    wheel \
    setuptools
fi

conda activate "${ENV_NAME}"

echo "Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

echo "Installing PyTorch target '${TORCH_TARGET}' from ${TORCH_REQUIREMENTS}..."
python -m pip install -r "${TORCH_REQUIREMENTS}"

echo "Installing medical-imaging and mesh packages from requirements.txt..."
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
PyTorch target: ${TORCH_TARGET}

Activate it with:
  conda activate ${ENV_NAME}

Start Jupyter with:
  jupyter lab

Expected project layout before running the notebook:
  models/checkpoint_segresnet.pth
  data/volumes_full/D1.nii.gz
  data/labels_full/D1.nii.gz
MSG
