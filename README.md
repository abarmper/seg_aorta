# SEG.A Aorta Lumen Inference

SegResNet-based aorta lumen segmentation on CT NIfTI files. Produces a predicted mask, surface meshes, 2D overlays, a slice-by-slice video, and an interactive 3D HTML. Optionally adds MC Dropout uncertainty maps and Seg-Grad-CAM saliency.

Two entry points, same pipeline:
- `demo.py` — CLI
- `SEGA_aorta_inference_mesh_tutorial.ipynb` — notebook

## Setup

```bash
bash setup_conda_environment.sh                     # CPU
TORCH_TARGET=cu118 bash setup_conda_environment.sh  # CUDA 11.8
ENV_NAME=seg-aorta bash setup_conda_environment.sh  # custom name
```

Then activate and launch Jupyter:

```bash
conda activate sega-aorta
jupyter lab
```

## Expected layout

```
project-root/
  models/checkpoint_segresnet.pth
  data/volumes_full/<CASE>.nii.gz
  data/labels_full/<CASE>.nii.gz
```

## Usage

```bash
python demo.py --case K18                     # full pipeline + explainability
python demo.py --case K18 --no-explain        # fast: segmentation only
python demo.py --case K18 --mc-passes 30      # more MC Dropout passes (default 20)
```

Outputs land in `outputs/<CASE>/`.

## Inference requirements (AWS)

See [`INFERENCE_REQUIREMENTS.md`](INFERENCE_REQUIREMENTS.md) for full details. Summary:

| Mode | GPU | VRAM | RAM | Time/case |
|------|-----|------|-----|-----------|
| Fast (`--no-explain`) | T4 / L4 16 GB | ≤ 6 GB | ~5 GB | 17–32 s |
| Explainability | A100 40/80 GB* | up to ~51 GB | ~7 GB | 53–99 s |

\* A10G/L4 24 GB works with automatic ROI fallback for Grad-CAM on large volumes.

## Notes

- `pymeshfix` is optional — mesh repair is skipped if unavailable.
- Video output tries MP4 first, falls back to GIF.
- CPU mode is supported via `--device cpu`.
- Python 3.10, PyTorch 2.0.1, MONAI 1.1.0 are the pinned target versions.
