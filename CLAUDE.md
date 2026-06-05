# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

SEG.A aorta lumen inference + visualization. A SegResNet (MONAI) checkpoint segments the aorta on a CT NIfTI, then meshes, an overlay PNG, an overlay video, and a side-by-side 3D HTML are produced. Two entry points share the same pipeline:

- `demo.py` — CLI: `python demo.py --case K18 [--project-root .] [--device cuda|cpu]`
- `SEGA_aorta_inference_mesh_tutorial.ipynb` — notebook version of the same flow.

There is no test suite, no linter config, and no package install — this is a single-script + notebook project.

## Required on-disk layout

The CLI/notebook expects, relative to `--project-root` (default `.`):

```
models/checkpoint_segresnet.pth
data/volumes_full/<CASE>.nii.gz
data/labels_full/<CASE>.nii.gz
```

Outputs land under `outputs/<CASE>/` (mask NIfTI, two OBJ meshes for pred + one for GT, best-slice PNG, MP4/GIF video, Plotly HTML).

## Environment

A pre-built venv lives at `.venv/`. All dependencies are pinned in a single `requirements.txt` (including the CUDA 12.1 torch wheels via `--extra-index-url`). To set up fresh, prefer the conda script over solving a single env (PyTorch + MONAI + SimpleITK + pymeshfix solves are slow):

```bash
bash setup_conda_environment.sh                    # default env name (sega-aorta)
ENV_NAME=seg-aorta bash setup_conda_environment.sh # custom env name
bash setup_environment.sh                          # venv (pip) instead of conda
```

Each script creates a minimal base env, then runs `pip install -r requirements.txt` and registers a Jupyter kernel. Into an existing Python 3.12 env, `pip install -r requirements.txt` alone is enough. For CPU-only, swap in the CPU torch wheels noted at the top of `requirements.txt`.

Pinned: Python 3.12, `torch==2.4.1+cu121`, `monai==1.4.0`, `numpy==1.26.4` (must stay `<2`), `scipy==1.13.1`. These are the exact versions in the validated `.venv` — do not bump without re-validating.

## Pipeline architecture (demo.py)

The flow in `main()` is linear and load-bearing; functions are not abstractions, they are stages:

1. **Model** — `build_segresnet()` constructs a specific SegResNet topology (`init_filters=8`, `blocks_down=(1,2,2,4)`, `blocks_up=(1,1,1)`, `UpsampleMode.DECONV`, 2 output classes). The checkpoint was trained against exactly this geometry; changing any of those args will break `load_state_dict(strict=True)`. `load_checkpoint` strips `module.` / `model.` prefixes and unwraps `state_dict` / `model_state_dict`.
2. **Preprocess** (`make_preprocess`) — MONAI dict transforms: load → channel-first → double `ScaleIntensityRanged` (HU clamp to ±3000, then window −275..1900 → 0..1) → `CropForegroundd` → RAS orientation → 1.0×1.0×1.5 mm spacing. The double scale is intentional (HU clamp before windowing).
3. **Inference** (`run_inference`) — `sliding_window_inference` with `ROI_SIZE=(160,160,160)`, `SW_BATCH_SIZE=2`, `overlap=0.25`. Then `Invertd` undoes preprocess transforms so the prediction lands back in the original image's geometry, followed by `AsDiscreted(argmax=True)`.
4. **Geometry bridge** — `prediction_to_sitk` handles both `xyz`-shaped and `zyx`-shaped pred tensors and copies the SimpleITK header from the original image. `vertices_array_to_physical` converts marching-cubes verts (in `zyx` index space) to physical-space `xyz` using the image's direction matrix and origin — this is what makes the OBJ meshes align with the source CT, do not "simplify" it.
5. **Meshing** (`create_meshes`) — marching cubes (skimage) → optional `pymeshfix` repair (best-effort, skipped if import fails) → Laplacian smoothing with `volume_constraint=True`. Returns `(smoothed, repaired)`. Smoothing failures fall back to the unsmoothed repaired mesh.
6. **Visualization** — `save_best_slice_png` picks the axial slice with the highest `gt + pred` voxel sum; `save_overlay_video` trims to the slab containing labels (±5 slice margin, step 2) and tries MP4 first, falling back to GIF; `save_3d_html` writes a Plotly side-by-side with `include_plotlyjs="cdn"`.

The notebook mirrors this same pipeline cell-for-cell — keep them in sync when changing logic.

## Conventions worth knowing

- Arrays are `zyx` (SimpleITK `GetArrayFromImage`) while SITK images, sizes, and spacings are `xyz` — `prediction_to_sitk` and `create_meshes` both transpose explicitly. Don't assume one convention.
- `pymeshfix` is optional at runtime; the `HAS_MESHFIX` flag at module import gates it. Don't make it a hard dependency.
- `torch.load(..., weights_only=True)` is used; checkpoints must be plain state dicts (not pickled objects).
- The repo is not a git repository.
