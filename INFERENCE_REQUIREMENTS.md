# Inference Requirements — SEG.A Aorta Lumen Segmentation

Benchmarks run on **2026-06-05**, GPU 0: NVIDIA A100-SXM4-80GB, driver 580.105.08, CUDA 13.0.  
Full end-to-end pipeline: model load → inference → meshing → overlay PNG → video (MP4) → 3D HTML.

---

## Benchmark Results

| Case | Mode | Wall time | Peak VRAM (allocated) | Peak VRAM (reserved) | Peak RAM (RSS) |
|------|------|----------:|----------------------:|---------------------:|---------------:|
| K18  | fast (`--no-explain`) | 31.6 s | 3.55 GB | 5.84 GB | 4.72 GB |
| K18  | explainability (20 MC passes + Seg-Grad-CAM) | 99.1 s | 43.4 GB | 51.0 GB | 6.80 GB |
| K19  | fast (`--no-explain`) | 17.4 s | 1.81 GB | 3.68 GB | 1.64 GB |
| K19  | explainability (20 MC passes + Seg-Grad-CAM) | 52.8 s | 7.09 GB | 8.19 GB | 3.17 GB |

> **Note on explainability VRAM:** Seg-Grad-CAM performs a full-volume forward+backward pass, so peak VRAM scales with the resampled volume extent. K18 (512×512×135 voxels, 0.75×0.75×5.0 mm) spiked to ~51 GB; K19 (512×512×99, 0.44×0.44×3.0 mm) used only ~8 GB. The code has a built-in CUDA OOM fallback to a centered 160³ ROI, so it will not crash on smaller GPUs — Grad-CAM quality degrades gracefully to a patch.

---

## Inference Requirements (AWS)

### Απαιτούμενη μνήμη RAM
- Observed peak: **~5–7 GB**
- Recommended: **≥ 16 GB RAM**

### Τύπος GPU που προτείνεται

| Use case | Recommended GPU |
|----------|----------------|
| Fast mode only (segmentation, no explainability) | NVIDIA **T4** or **L4** (16 GB) |
| With explainability, best-effort Grad-CAM | NVIDIA **A10G** or **L4** (24 GB) |
| With explainability, guaranteed full-volume Grad-CAM | NVIDIA **A100** (40/80 GB) |

### Απαιτούμενη μνήμη GPU (VRAM)

- **Fast mode:** ≤ 6 GB — a T4/L4 16 GB is more than sufficient.
- **Explainability mode:** up to ~51 GB on larger volumes (full-volume Grad-CAM). On GPUs with less VRAM the pipeline automatically falls back to a 160³ ROI for Grad-CAM; ~24 GB handles all tested cases with fallback.

### Χώρος αποθήκευσης

| Item | Size |
|------|------|
| Model checkpoint (`checkpoint_segresnet.pth`) | 4.6 MB |
| Model + source code | ~4.8 MB |
| Full Python environment (venv, PyTorch + MONAI + CUDA) | ~5.8 GB |
| Output artifacts per case — fast mode | a few MB |
| Output artifacts per case — explainability (NIfTIs + MP4 + HTML) | ~450 MB |

### Ενδεικτικός χρόνος inference ανά εξέταση

- **Fast mode:** ~17–32 s/case (full pipeline including meshing and video rendering)
- **Explainability mode (20 MC passes):** ~53–99 s/case

> Part of the wall time is CPU-bound (marching cubes, Laplacian smoothing, ffmpeg video encoding, Plotly HTML). Pure GPU inference is a subset of these numbers.

### Πρόσθετες εξαρτήσεις / απαιτήσεις λογισμικού

- CUDA-capable GPU with NVIDIA driver (tested: driver 580.105.08, CUDA runtime 13.0, PyTorch built against cu121)
- `ffmpeg` — provided via `imageio-ffmpeg` for MP4 output
- `pymeshfix` — optional; enables best-effort mesh repair (pipeline continues without it)
- CPU-only mode is supported via `--device cpu` (significantly slower)

---

## Software Environment

| Component | Pinned (requirements.txt) | Installed / tested |
|-----------|--------------------------|-------------------|
| Python | 3.10 | **3.12.3** |
| PyTorch | 2.0.1+cu118 | **2.4.1+cu121** |
| MONAI | 1.1.0 | **1.4.0** |
| NumPy | <2 | 1.26.4 |
| SciPy | <1.12 | 1.13.1 |
| SimpleITK | ≥2.2, <2.4 | installed |
| scikit-image | ≥0.21, <0.23 | installed |
| trimesh | ≥3.22, <4.1 | installed |
| matplotlib | ≥3.7, <3.9 | installed |
| plotly | ≥5.18, <6 | installed |
| imageio / imageio-ffmpeg | per requirements.txt | installed |
| pymeshfix | ≥0.16, <0.18 | installed |

> ⚠️ The `requirements.txt` pins (`monai==1.1.0`, `torch==2.0.1`, `numpy<2`, `scipy<1.12`) reflect the originally validated stack. The benchmarks above were run on the **installed** environment (torch 2.4.1 / monai 1.4.0 / Python 3.12). For reproducible deployment, use the pinned versions via `setup_conda_environment.sh`.

### Environment setup

```bash
# CPU
bash setup_conda_environment.sh

# CUDA 11.8
TORCH_TARGET=cu118 bash setup_conda_environment.sh

# Custom env name
ENV_NAME=seg-aorta bash setup_conda_environment.sh
```

Dependency files: `requirements.txt`, `requirements-torch-cu118.txt`, `requirements-torch-cpu.txt`, `environment.yml`.
