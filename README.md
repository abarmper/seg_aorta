# SEG.A Aorta Lumen Inference

SegResNet-based aorta lumen segmentation on CT NIfTI files. For one case it runs inference and produces a predicted mask, surface meshes (smoothed + volume), a 2D CT/GT/Pred overlay, a slice-by-slice video, and an interactive side-by-side 3D HTML. By default it also adds explainability artifacts: MC Dropout uncertainty maps and Seg-Grad-CAM saliency.

Two entry points share the exact same pipeline:
- `demo.py` — command-line interface
- `SEGA_aorta_inference_mesh_tutorial.ipynb` — notebook walkthrough

## Quick start

A pre-built virtual environment is included at `.venv/`. Activate it and run the demo directly:

```bash
source .venv/bin/activate
python demo.py --case K18
```

Or run without activating:

```bash
.venv/bin/python demo.py --case K18
```

## Expected layout

Place your files relative to the project root (`--project-root`, default `.`):

```
project-root/
  models/checkpoint_segresnet.pth
  data/volumes_full/<CASE>.nii.gz
  data/labels_full/<CASE>.nii.gz
```

Outputs are written to `outputs/<CASE>/`.

## Running the demo

```bash
python demo.py --case K18                       # full pipeline + explainability
python demo.py --case K18 --no-explain          # fast: segmentation only (alias: --fast)
python demo.py --case K18 --device cuda:0        # pick a specific GPU
python demo.py --case K18 --device cpu           # run on CPU
python demo.py --case K18 --mc-passes 30         # more MC Dropout passes (default 20)
python demo.py --case K18 --gradcam-layer up_layers.-1   # Grad-CAM target layer
python demo.py --case K18 --project-root /path/to/root   # use a different root
```

| Flag | Default | Description |
|------|---------|-------------|
| `--case` | `K18` | Case name; expects matching files under `data/`. |
| `--project-root` | `.` | Root containing `models/`, `data/`, `outputs/`. |
| `--device` | auto | `cuda`, `cuda:N`, or `cpu`. Defaults to CUDA if available. |
| `--no-explain` / `--fast` | off | Skip all explainability; single deterministic pass. |
| `--mc-passes` | `20` | Number of MC Dropout stochastic passes. |
| `--gradcam-layer` | `up_layers.-1` | Dotted path to the Grad-CAM target layer. |

### Outputs

In every mode (`outputs/<CASE>/`):
- `<CASE>_predicted_aorta_lumen.nii.gz` — predicted mask
- `<CASE>_aortic_vessel_tree_smoothed.obj`, `..._volume_mesh.obj` — prediction meshes
- `<CASE>_ground_truth_aortic_vessel_tree_smoothed.obj` — GT mesh
- `<CASE>_ct_gt_pred_overlay_best_slice.png`, `..._overlay_video.mp4` — 2D overlays
- `<CASE>_gt_pred_3d_comparison.html` — interactive 3D comparison

Additionally with explainability (default, omitted under `--no-explain`):
- `<CASE>_uncertainty_{mean_prob,entropy,std}.nii.gz` + entropy overlay PNG/video
- `<CASE>_seggradcam.nii.gz` + saliency overlay PNG/video
- the 3D HTML's predicted mesh is colored per-vertex by entropy and saliency

## Notebook

```bash
source .venv/bin/activate
jupyter lab        # then open SEGA_aorta_inference_mesh_tutorial.ipynb
```

The notebook mirrors `demo.py` cell-for-cell.

## Setting up the environment from scratch

If you need to rebuild the environment, two scripts are provided.

**venv (pip):**

```bash
bash setup_environment.sh                              # CPU
TORCH_TARGET=cu118 bash setup_environment.sh           # CUDA 11.8
PYTHON_BIN=python3.10 VENV_DIR=.venv bash setup_environment.sh
```

**conda** (recommended for fresh machines — avoids slow PyTorch/MONAI/pymeshfix solves):

```bash
bash setup_conda_environment.sh                        # CPU
TORCH_TARGET=cu118 bash setup_conda_environment.sh     # CUDA 11.8
ENV_NAME=seg-aorta bash setup_conda_environment.sh     # custom env name
```

Both create a minimal base environment, then pip-install torch from
`requirements-torch-{cpu,cu118}.txt` and the medical/mesh stack from
`requirements.txt`, and register a Jupyter kernel.

## Inference requirements (AWS)

See [`INFERENCE_REQUIREMENTS.md`](INFERENCE_REQUIREMENTS.md) for full benchmark details. Summary:

| Mode | GPU | VRAM | RAM | Time/case |
|------|-----|------|-----|-----------|
| Fast (`--no-explain`) | T4 / L4 16 GB | ≤ 6 GB | ~5 GB | 17–32 s |
| Explainability | A100 40/80 GB* | up to ~51 GB | ~7 GB | 53–99 s |

\* A10G / L4 24 GB also works, relying on the automatic ROI fallback for Grad-CAM on large volumes.

## Notes

- `pymeshfix` is optional — mesh repair is skipped if it cannot be imported.
- Video output tries MP4 first and falls back to GIF if encoding is unavailable.
- CPU mode is supported via `--device cpu` (slower).
- The pinned target versions are Python 3.10, PyTorch 2.0.1, MONAI 1.1.0. The included `.venv/` runs Python 3.12 / PyTorch 2.4.1 / MONAI 1.4.0; see `INFERENCE_REQUIREMENTS.md` for the full comparison.
