# SEG.A Aorta Lumen Inference

SegResNet-based aorta lumen segmentation on CT NIfTI files. For one case it runs inference and produces a predicted mask, surface meshes (smoothed + volume), a 2D CT/GT/Pred overlay, a slice-by-slice video, and an interactive side-by-side 3D HTML. By default it also adds explainability artifacts: MC Dropout uncertainty maps and Seg-Grad-CAM saliency.

Two entry points share the exact same pipeline:
- `demo.py` — command-line interface
- `SEGA_aorta_inference_mesh_tutorial.ipynb` — notebook walkthrough

## Quick start

Set up the environment first (conda recommended; see [Setting up the environment](#setting-up-the-environment-from-scratch) for options):

```bash
bash setup_conda_environment.sh
```

Then activate and run:

```bash
conda activate sega-aorta
python demo.py --case K18
```

All dependencies — including the CUDA 12.1 PyTorch build — are pinned in a single
`requirements.txt`, so `pip install -r requirements.txt` is enough on an existing
Python 3.12 environment.

## Expected layout

Place your files relative to the project root (`--project-root`, default `.`):

```
project-root/
  models/checkpoint_segresnet.pth
  data/volumes_full/<CASE>.nii.gz
  data/labels_full/<CASE>.nii.gz   # optional — see label-free mode below
```

The label is optional. If `data/labels_full/<CASE>.nii.gz` is missing, or you pass
`--no-label`/`--test`, the demo runs in **label-free (test) mode**: it segments the
volume and skips everything that needs a ground truth (Dice, GT mesh, GT overlay/panel).

Outputs are written to `outputs/<CASE>/`.

## Running the demo

```bash
python demo.py --case K18                       # full pipeline + explainability
python demo.py --case K18 --no-explain          # fast: segmentation only (alias: --fast)
python demo.py --case K18 --no-label            # label-free / test mode (alias: --test)
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
| `--no-label` / `--test` | off | Label-free mode: skip Dice, GT mesh, and GT overlay/panel. Auto-engaged if the label file is absent. |
| `--mc-passes` | `20` | Number of MC Dropout stochastic passes. |
| `--gradcam-layer` | `up_layers.-1` | Dotted path to the Grad-CAM target layer. |

### Outputs

In every mode (`outputs/<CASE>/`):
- `<CASE>_predicted_aorta_lumen.nii.gz` — predicted mask
- `<CASE>_aortic_vessel_tree_smoothed.obj`, `..._volume_mesh.obj` — prediction meshes
- `<CASE>_ground_truth_aortic_vessel_tree_smoothed.obj` — GT mesh *(skipped in label-free mode)*
- `<CASE>_ct_gt_pred_overlay_best_slice.png`, `..._overlay_video.mp4` — 2D overlays *(prediction-only in label-free mode)*
- `<CASE>_gt_pred_3d_comparison.html` — interactive 3D comparison *(single PRED panel in label-free mode)*

Additionally with explainability (default, omitted under `--no-explain`):
- `<CASE>_uncertainty_{mean_prob,entropy,std}.nii.gz` + entropy overlay PNG/video
- `<CASE>_seggradcam.nii.gz` + saliency overlay PNG/video
- the 3D HTML's predicted mesh is colored per-vertex by entropy and saliency

## Notebook

```bash
conda activate sega-aorta
jupyter lab        # then open SEGA_aorta_inference_mesh_tutorial.ipynb
```

The notebook mirrors `demo.py` cell-for-cell.

## Setting up the environment from scratch

All dependencies live in a single fully-pinned `requirements.txt` (Python 3.12,
`torch==2.4.1+cu121`, `monai==1.4.0`). If you need to rebuild the environment,
two scripts are provided.

**venv (pip):**

```bash
bash setup_environment.sh
PYTHON_BIN=python3.12 VENV_DIR=.venv bash setup_environment.sh
```

**conda** (recommended for fresh machines — avoids slow PyTorch/MONAI/pymeshfix solves):

```bash
bash setup_conda_environment.sh
ENV_NAME=seg-aorta bash setup_conda_environment.sh     # custom env name
```

Both create a minimal base environment, then `pip install -r requirements.txt`
(which includes the CUDA 12.1 torch wheels) and register a Jupyter kernel. For a
CPU-only machine, install the CPU torch wheels as noted at the top of
`requirements.txt`.

## Notes

- `pymeshfix` is optional — mesh repair is skipped if it cannot be imported.
- Video output tries MP4 first and falls back to GIF if encoding is unavailable.
- CPU mode is supported via `--device cpu` (slower).
- The pinned target versions are Python 3.12, PyTorch 2.4.1+cu121, MONAI 1.4.0 (see `requirements.txt`).
