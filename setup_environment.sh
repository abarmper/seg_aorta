#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash setup_environment.sh                         # CPU environment
#   TORCH_TARGET=cu118 bash setup_environment.sh      # CUDA 11.8 PyTorch wheels
#   PYTHON_BIN=python3.10 VENV_DIR=.venv-sega bash setup_environment.sh

TORCH_TARGET="${TORCH_TARGET:-cpu}"   # cpu | cu118 | pypi
VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
KERNEL_NAME="${KERNEL_NAME:-sega-aorta}"
KERNEL_DISPLAY_NAME="${KERNEL_DISPLAY_NAME:-Python (sega-aorta)}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "${PYTHON_BIN} was not found; falling back to python3."
  PYTHON_BIN="python3"
fi

"${PYTHON_BIN}" - <<'PY'
import sys
major, minor = sys.version_info[:2]
if (major, minor) < (3, 8) or (major, minor) > (3, 10):
    print(f"WARNING: Python {major}.{minor} detected. This tutorial is best tested with Python 3.10 because it uses torch==2.0.1 and monai==1.1.0.")
PY

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

case "${TORCH_TARGET}" in
  cpu)
    python -m pip install -r requirements-torch-cpu.txt
    ;;
  cu118)
    python -m pip install -r requirements-torch-cu118.txt
    ;;
  pypi)
    python -m pip install torch==2.0.1 torchvision==0.15.2
    ;;
  *)
    echo "Unknown TORCH_TARGET='${TORCH_TARGET}'. Use cpu, cu118, or pypi."
    exit 1
    ;;
esac

python -m pip install -r requirements.txt
python -m ipykernel install --user --name "${KERNEL_NAME}" --display-name "${KERNEL_DISPLAY_NAME}"

cat <<EOF

Environment created at: ${VENV_DIR}
Notebook kernel: ${KERNEL_DISPLAY_NAME}

Activate it with:
  source ${VENV_DIR}/bin/activate

Start Jupyter with:
  jupyter lab

Expected project layout before running the notebook:
  models/checkpoint_segresnet.pth
  data/volumes_full/D1.nii.gz
  data/labels_full/D1.nii.gz
EOF
