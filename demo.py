"""SEG.A aorta lumen inference + demo.

Runs SegResNet inference on a single case and produces:
  - predicted mask (NIfTI)
  - smoothed and volume meshes (OBJ)
  - 2D CT/GT/Pred overlay (PNG)
  - slice-by-slice overlay video (MP4, with GIF fallback)
  - side-by-side 3D HTML (Plotly)

Usage:
  python demo.py --case K18
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import SimpleITK as sitk
import torch
import trimesh
from matplotlib.patches import Patch
from plotly.subplots import make_subplots
from skimage import measure

from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet
from monai.transforms import (
    AsDiscreted,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    Invertd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)
from monai.utils.enums import UpsampleMode

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import pymeshfix
    HAS_MESHFIX = True
except Exception as exc:
    HAS_MESHFIX = False
    print(f"pymeshfix not available, repair will be skipped. Reason: {exc}")


ROI_SIZE = (160, 160, 160)
SW_BATCH_SIZE = 2


def build_segresnet() -> SegResNet:
    return SegResNet(
        spatial_dims=3,
        init_filters=8,
        in_channels=1,
        out_channels=2,
        dropout_prob=0.35,
        num_groups=8,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        upsample_mode=UpsampleMode.DECONV,
    )


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
    if isinstance(ckpt, dict):
        ckpt = {k.replace("module.", "").replace("model.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=True)
    return model.to(device).eval()


def make_preprocess() -> Compose:
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityRanged(keys=["image"], a_min=-3000, a_max=3000, b_min=-3000, b_max=3000, clip=True),
        ScaleIntensityRanged(keys=["image"], a_min=-275, a_max=1900, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image"], source_key="image"),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.5), mode="bilinear"),
    ])


def make_postprocess(preprocess: Compose) -> Compose:
    return Compose([
        Invertd(
            keys="pred",
            transform=preprocess,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=True),
    ])


def prediction_to_sitk(pred_tensor: torch.Tensor, reference: sitk.Image) -> sitk.Image:
    pred = np.squeeze(pred_tensor.detach().cpu().numpy().astype(np.uint8))
    size_xyz = tuple(reference.GetSize())
    if tuple(pred.shape) == size_xyz:
        pred_zyx = np.transpose(pred, (2, 1, 0))
    elif tuple(pred.shape) == size_xyz[::-1]:
        pred_zyx = pred
    else:
        raise ValueError(f"Pred shape {pred.shape} vs reference xyz {size_xyz}")
    img = sitk.GetImageFromArray(pred_zyx)
    img.CopyInformation(reference)
    return sitk.Cast(img, sitk.sitkUInt8)


def run_inference(image_path: Path, model: torch.nn.Module, device: torch.device) -> sitk.Image:
    preprocess = make_preprocess()
    postprocess = make_postprocess(preprocess)
    reference = sitk.ReadImage(str(image_path))

    dataset = Dataset(data=[{"image": str(image_path)}], transform=preprocess)
    loader = DataLoader(dataset, batch_size=1, num_workers=0)

    with torch.no_grad():
        batch = next(iter(loader))
        inputs = batch["image"].to(device)
        logits = sliding_window_inference(
            inputs=inputs,
            roi_size=ROI_SIZE,
            sw_batch_size=SW_BATCH_SIZE,
            predictor=model,
            overlap=0.25,
        )
        batch["pred"] = logits
        restored = [postprocess(item) for item in decollate_batch(batch)]
        pred_tensor = restored[0]["pred"]
    return prediction_to_sitk(pred_tensor, reference)


def vertices_array_to_physical(verts_zyx: np.ndarray, image: sitk.Image) -> np.ndarray:
    verts_xyz = verts_zyx[:, ::-1]
    direction = np.array(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    origin = np.array(image.GetOrigin(), dtype=np.float64)
    return origin + verts_xyz @ direction.T


def create_meshes(mask_image: sitk.Image, smoothing_iterations: int = 10):
    mask_zyx = sitk.GetArrayFromImage(mask_image).astype(np.uint8)
    if mask_zyx.max() == 0:
        raise ValueError("Empty mask, marching cubes cannot run.")
    padded = np.pad(mask_zyx, 1, mode="constant", constant_values=0)
    spacing_zyx = np.array(mask_image.GetSpacing()[::-1], dtype=np.float64)
    verts_zyx, faces, _, _ = measure.marching_cubes(padded, level=0.5, spacing=spacing_zyx)
    verts_zyx -= spacing_zyx
    verts_physical = vertices_array_to_physical(verts_zyx, mask_image)
    base = trimesh.Trimesh(vertices=verts_physical, faces=faces, process=False)

    if HAS_MESHFIX:
        try:
            mf = pymeshfix.MeshFix(base.vertices, base.faces)
            mf.repair()
            repaired = trimesh.Trimesh(vertices=mf.points, faces=mf.faces, process=False)
        except Exception as exc:
            print(f"Mesh repair failed: {exc}; using raw mesh")
            repaired = base
    else:
        repaired = base

    smoothed = repaired.copy()
    try:
        result = trimesh.smoothing.filter_laplacian(
            smoothed, lamb=0.5, iterations=smoothing_iterations,
            implicit_time_integration=False, volume_constraint=True,
        )
        if isinstance(result, trimesh.Trimesh):
            smoothed = result
    except Exception as exc:
        print(f"Mesh smoothing failed: {exc}")
        smoothed = repaired.copy()
    return smoothed, repaired


def dice_score(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return 2.0 * np.logical_and(a, b).sum() / denom


def normalize_for_display(slice_2d: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(slice_2d, [1, 99])
    if hi <= lo:
        return slice_2d
    return np.clip((slice_2d - lo) / (hi - lo), 0, 1)


def choose_best_axial_slice(gt: np.ndarray, pred: np.ndarray) -> int:
    score = gt.sum(axis=(1, 2)) + pred.sum(axis=(1, 2))
    if score.max() == 0:
        return gt.shape[0] // 2
    return int(np.argmax(score))


def make_label_overlay_rgba(gt: np.ndarray, pred: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    gt = gt.astype(bool); pred = pred.astype(bool)
    rgba = np.zeros((*gt.shape, 4), dtype=np.float32)
    rgba[gt & ~pred] = (0.0, 1.0, 0.0, alpha)
    rgba[pred & ~gt] = (1.0, 0.0, 1.0, alpha)
    rgba[gt & pred] = (1.0, 1.0, 0.0, alpha)
    return rgba


def save_best_slice_png(image: np.ndarray, gt: np.ndarray, pred: np.ndarray, case_id: str, out_path: Path):
    z = choose_best_axial_slice(gt, pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(normalize_for_display(image[z]), cmap="gray")
    ax.imshow(make_label_overlay_rgba(gt[z], pred[z]))
    ax.set_title(f"{case_id}: CT + GT + Pred | z={z} | Dice={dice_score(gt, pred):.4f}", fontsize=14)
    ax.axis("off")
    handles = [
        Patch(facecolor=(0.0, 1.0, 0.0, 0.55), label="GT only"),
        Patch(facecolor=(1.0, 0.0, 1.0, 0.55), label="Prediction only"),
        Patch(facecolor=(1.0, 1.0, 0.0, 0.55), label="GT ∩ Pred"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved overlay PNG: {out_path}")


def render_overlay_frame(image, gt, pred, z, case_id, overall_dice, dpi=120):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    ax.imshow(normalize_for_display(image[z]), cmap="gray")
    ax.imshow(make_label_overlay_rgba(gt[z], pred[z]))
    ax.set_title(f"{case_id} | axial z={z} | Dice={overall_dice:.4f}")
    ax.axis("off")
    plt.tight_layout(pad=0.2)
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def save_overlay_video(image, gt, pred, case_id, mp4_path: Path, gif_path: Path, fps=12, step=2, margin=5):
    combined = gt | pred
    counts = combined.sum(axis=(1, 2))
    nonzero = np.flatnonzero(counts)
    if len(nonzero) == 0:
        slices = list(range(0, gt.shape[0], step))
    else:
        start = max(0, int(nonzero[0]) - margin)
        stop = min(gt.shape[0], int(nonzero[-1]) + margin + 1)
        slices = list(range(start, stop, step))

    print(f"Rendering {len(slices)} video frames...")
    overall_dice = dice_score(gt, pred)
    frames = [render_overlay_frame(image, gt, pred, z, case_id, overall_dice) for z in slices]

    try:
        imageio.mimsave(mp4_path, frames, fps=fps, codec="libx264", macro_block_size=1)
        print(f"Saved MP4: {mp4_path}")
        return mp4_path
    except Exception as exc:
        print(f"MP4 failed: {exc}; writing GIF instead")
        imageio.mimsave(gif_path, frames, duration=1.0 / fps)
        print(f"Saved GIF: {gif_path}")
        return gif_path


def mesh_to_trace(mesh: trimesh.Trimesh, name: str, color: str = "red") -> go.Mesh3d:
    v = np.asarray(mesh.vertices); f = np.asarray(mesh.faces)
    return go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        name=name, color=color, opacity=1.0, flatshading=False,
        lighting=dict(ambient=0.25, diffuse=0.75, specular=0.35, roughness=0.55, fresnel=0.2),
        lightposition=dict(x=100, y=200, z=300),
        showscale=False,
    )


def save_3d_html(gt_mesh: trimesh.Trimesh, pred_mesh: trimesh.Trimesh, case_id: str, out_path: Path):
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                        subplot_titles=("GT", "PRED"), horizontal_spacing=0.02)
    fig.add_trace(mesh_to_trace(gt_mesh, "GT"), row=1, col=1)
    fig.add_trace(mesh_to_trace(pred_mesh, "PRED"), row=1, col=2)
    for col in (1, 2):
        fig.update_scenes(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode="data", camera=dict(eye=dict(x=1.4, y=-2.0, z=0.9)),
            row=1, col=col,
        )
    fig.update_layout(
        title=f"{case_id}: ground truth vs prediction 3D aorta mesh",
        width=1000, height=900, showlegend=False,
        margin=dict(l=0, r=0, t=80, b=0),
        paper_bgcolor="white", plot_bgcolor="white", font=dict(size=18),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"Saved 3D HTML: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="K18")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--device", default=None, help="cuda, cuda:N, or cpu")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    case = args.case

    image_path = root / "data" / "volumes_full" / f"{case}.nii.gz"
    gt_path = root / "data" / "labels_full" / f"{case}.nii.gz"
    model_path = root / "models" / "checkpoint_segresnet.pth"
    out_dir = root / "outputs" / case
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in (image_path, gt_path, model_path):
        if not p.exists():
            raise FileNotFoundError(p)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    print("Building model + loading checkpoint...")
    model = build_segresnet()
    model = load_checkpoint(model, model_path, device)

    print(f"Running inference on {image_path.name}...")
    pred_sitk = run_inference(image_path, model, device)
    pred_mask_path = out_dir / f"{case}_predicted_aorta_lumen.nii.gz"
    sitk.WriteImage(pred_sitk, str(pred_mask_path), True)
    print(f"Saved predicted mask: {pred_mask_path}")
    print(f"  size xyz: {pred_sitk.GetSize()}  spacing xyz: {pred_sitk.GetSpacing()}")

    image_sitk = sitk.ReadImage(str(image_path))
    gt_sitk = sitk.ReadImage(str(gt_path))
    image_zyx = sitk.GetArrayFromImage(image_sitk).astype(np.float32)
    gt_zyx = sitk.GetArrayFromImage(gt_sitk) > 0
    pred_zyx = sitk.GetArrayFromImage(pred_sitk) > 0
    print(f"Dice: {dice_score(gt_zyx, pred_zyx):.4f}")

    print("Creating prediction meshes...")
    pred_smoothed, pred_volume = create_meshes(pred_sitk, smoothing_iterations=10)
    pred_smoothed.export(str(out_dir / f"{case}_aortic_vessel_tree_smoothed.obj"))
    pred_volume.export(str(out_dir / f"{case}_aortic_vessel_tree_volume_mesh.obj"))
    print(f"  smoothed: {len(pred_smoothed.vertices):,} verts / {len(pred_smoothed.faces):,} faces")

    print("Creating GT meshes...")
    gt_smoothed, _ = create_meshes(gt_sitk, smoothing_iterations=10)
    gt_smoothed.export(str(out_dir / f"{case}_ground_truth_aortic_vessel_tree_smoothed.obj"))

    print("Creating 2D overlay PNG...")
    save_best_slice_png(image_zyx, gt_zyx, pred_zyx, case, out_dir / f"{case}_ct_gt_pred_overlay_best_slice.png")

    print("Creating overlay video...")
    save_overlay_video(
        image_zyx, gt_zyx, pred_zyx, case,
        mp4_path=out_dir / f"{case}_ct_gt_pred_overlay_video.mp4",
        gif_path=out_dir / f"{case}_ct_gt_pred_overlay_video.gif",
        fps=12, step=2,
    )

    print("Creating 3D HTML...")
    save_3d_html(gt_smoothed, pred_smoothed, case, out_dir / f"{case}_gt_pred_3d_comparison.html")

    print(f"\nAll outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
