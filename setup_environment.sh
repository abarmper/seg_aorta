#!/usr/bin/env bash
set -euo pipefail

# Create the venv used for SEG.A aorta inference.
#
# Usage:
#   bash setup_environment.sh
#   PYTHON_BIN=python3.12 VENV_DIR=.venv bash setup_environment.sh
#
# All dependencies (including CUDA 12.1 torch) are pinned in requirements.txt,
# so this script just builds a venv and installs that one file.

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
KERNEL_NAME="${KERNEL_NAME:-sega-aorta}"
KERNEL_DISPLAY_NAME="${KERNEL_DISPLAY_NAME:-Python (sega-aorta)}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "${PYTHON_BIN} was not found; falling back to python3."
  PYTHON_BIN="python3"
fi

"${PYTHON_BIN}" - <<'PY'
import sys
major, minor = sys.version_info[:2]
if (major, minor) != (3, 12):
    print(f"WARNING: Python {major}.{minor} detected. The validated stack uses Python 3.12.")
PY

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
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
