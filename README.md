# SEG.A Aorta Lumen Inference + Mesh Tutorial

This package contains a Jupyter notebook tutorial for running a SegResNet-based aorta lumen segmentation model on NIfTI files, creating meshes, visualizing the CT image with ground truth and prediction overlaid in different colors, creating a slice-by-slice video, and rendering a side-by-side 3D aorta comparison.

## Files

- `SEGA_aorta_inference_mesh_tutorial.ipynb` — main tutorial notebook.
- `setup_conda_environment.sh` — recommended Conda setup script.
- `environment.yml` — minimal CPU Conda base environment.
- `environment-cu118.yml` — minimal CUDA Conda base environment.
- `requirements.txt` — medical-imaging, mesh, and plotting packages installed with pip.
- `requirements-torch-cpu.txt` — CPU PyTorch wheels.
- `requirements-torch-cu118.txt` — CUDA 11.8 PyTorch wheels.

## Expected project layout

Before running the notebook, place your files like this:

```text
project-root/
  SEGA_aorta_inference_mesh_tutorial.ipynb
  models/
    checkpoint_segresnet.pth
  data/
    volumes_full/
      D1.nii.gz
    labels_full/
      D1.nii.gz
```

## Recommended setup with Conda

From this package folder, run:

```bash
bash setup_conda_environment.sh
```

For an NVIDIA CUDA 11.8 PyTorch environment, run:

```bash
TORCH_TARGET=cu118 bash setup_conda_environment.sh
```

To use a custom environment name:

```bash
ENV_NAME=seg-aorta bash setup_conda_environment.sh
```

Then activate and launch Jupyter:

```bash
conda activate sega-aorta
jupyter lab
```

If you used a custom environment name, activate that name instead.

## Why the setup script installs most packages with pip

Solving a full Conda environment with PyTorch, MONAI, SimpleITK, scikit-image, and pymeshfix can be slow. The setup script creates a small Conda environment first, then installs the medical-imaging stack with pip. This usually avoids long `Solving environment` hangs.

## Notebook outputs

For case `D1`, outputs are written to:

```text
outputs/D1/
  D1_predicted_aorta_lumen.nii.gz
  D1_aortic_vessel_tree_volume_mesh.obj
  D1_aortic_vessel_tree_smoothed.obj
  D1_ground_truth_aortic_vessel_tree_smoothed.obj
  D1_ct_gt_pred_overlay_best_slice.png
  D1_ct_gt_pred_overlay_video.mp4
  D1_gt_pred_3d_comparison.html
```

`pymeshfix` is optional at runtime. If it cannot be imported on your platform, the notebook still creates a marching-cubes mesh and skips the repair step. The video cell writes MP4 first and falls back to GIF if MP4 encoding is unavailable. The 3D comparison is saved as an interactive Plotly HTML file.
