"""SEG.A aorta lumen inference + explainability demo.

For one case, runs SegResNet inference and produces:
  - predicted mask (NIfTI)
  - smoothed and volume meshes (OBJ)
  - 2D CT/GT/Pred overlay (PNG) and slice-by-slice video (MP4/GIF)
  - side-by-side 3D HTML (Plotly); when explainability is on, the predicted
    aorta surface is additionally rendered colored by entropy and Seg-Grad-CAM

And, unless --no-explain is set, the explainability artifacts:
  - MC Dropout: mean foreground probability, predictive entropy, std (NIfTI)
              + entropy overlay PNG and video
  - Seg-Grad-CAM: saliency volume (NIfTI) + overlay PNG and video
  - the 3D HTML's predicted mesh is colored per-vertex by entropy and saliency

All artifacts land in outputs/<case>/.

Usage:
  python demo.py --case K18
  python demo.py --case K18 --fast                        # fast: skip all explainability
  python demo.py --case K18 --no-explain                  # same as --fast
  python demo.py --case K18 --mc-passes 30                # more dropout passes
  python demo.py --case K18 --gradcam-layer up_layers.-1  # default target layer
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from sega import (
    SegGradCAM,
    build_segresnet,
    create_meshes,
    load_checkpoint,
    run_inference,
    run_mc_inference,
    viz,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="K18")
    p.add_argument("--project-root", default=".")
    p.add_argument("--device", default=None, help="cuda, cuda:N, or cpu")
    p.add_argument("--no-label", "--test", dest="no_label", action="store_true",
                   help="Label-free (test) mode: run inference without a ground-truth label. "
                        "Skips Dice, the GT mesh, and the GT overlay/panel. Also engaged "
                        "automatically when the label file is absent.")
    p.add_argument("--no-explain", "--fast", dest="no_explain", action="store_true",
                   help="Fast mode: skip all explainability (MC Dropout uncertainty + Seg-Grad-CAM). "
                        "Runs a single deterministic pass instead of --mc-passes, and produces no "
                        "uncertainty/saliency NIfTIs, heatmaps, or videos.")
    p.add_argument("--mc-passes", type=int, default=20,
                   help="Number of MC Dropout stochastic passes (default 20).")
    p.add_argument("--gradcam-layer", default="up_layers.-1",
                   help="Dotted attribute path of the Grad-CAM target layer on the model "
                        "(default: up_layers.-1, the last decoder ResBlock before conv_final).")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.project_root).resolve()
    case = args.case

    image_path = root / "data" / "volumes_full" / f"{case}.nii.gz"
    gt_path = root / "data" / "labels_full" / f"{case}.nii.gz"
    model_path = root / "models" / "checkpoint_segresnet.pth"
    out_dir = root / "outputs" / case
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in (image_path, model_path):
        if not p.exists():
            raise FileNotFoundError(p)

    has_gt = gt_path.exists() and not args.no_label
    if not has_gt:
        reason = "--no-label/--test set" if args.no_label else f"label not found: {gt_path}"
        print(f"Label-free (test) mode: {reason}")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    print("Building model + loading checkpoint...")
    model = build_segresnet()
    model = load_checkpoint(model, model_path, device)

    image_sitk = sitk.ReadImage(str(image_path))
    image_zyx = sitk.GetArrayFromImage(image_sitk).astype(np.float32)
    if has_gt:
        gt_sitk = sitk.ReadImage(str(gt_path))
        gt_zyx = sitk.GetArrayFromImage(gt_sitk) > 0
    else:
        gt_sitk = None

    # ----- inference -----
    if args.no_explain:
        print(f"Running deterministic inference on {image_path.name}...")
        pred_sitk = run_inference(image_path, model, device)
        uncertainty = None
    else:
        print(f"Running MC Dropout inference ({args.mc_passes} passes) on {image_path.name}...")
        uncertainty = run_mc_inference(image_path, model, device, n_passes=args.mc_passes)
        pred_sitk = uncertainty["pred"]

    pred_mask_path = out_dir / f"{case}_predicted_aorta_lumen.nii.gz"
    sitk.WriteImage(pred_sitk, str(pred_mask_path), True)
    print(f"Saved predicted mask: {pred_mask_path}")
    print(f"  size xyz: {pred_sitk.GetSize()}  spacing xyz: {pred_sitk.GetSpacing()}")

    pred_zyx = sitk.GetArrayFromImage(pred_sitk) > 0
    if not has_gt:
        gt_zyx = np.zeros_like(pred_zyx, dtype=bool)
    if has_gt:
        print(f"Dice: {viz.dice_score(gt_zyx, pred_zyx):.4f}")

    # ----- meshes -----
    print("Creating prediction meshes...")
    pred_smoothed, pred_volume = create_meshes(pred_sitk, smoothing_iterations=10)
    pred_smoothed.export(str(out_dir / f"{case}_aortic_vessel_tree_smoothed.obj"))
    pred_volume.export(str(out_dir / f"{case}_aortic_vessel_tree_volume_mesh.obj"))
    print(f"  smoothed: {len(pred_smoothed.vertices):,} verts / {len(pred_smoothed.faces):,} faces")

    if has_gt:
        print("Creating GT meshes...")
        gt_smoothed, _ = create_meshes(gt_sitk, smoothing_iterations=10)
        gt_smoothed.export(str(out_dir / f"{case}_ground_truth_aortic_vessel_tree_smoothed.obj"))
    else:
        gt_smoothed = None

    # ----- overlay PNG + video + 3D HTML -----
    print("Creating 2D overlay PNG...")
    viz.save_best_slice_png(image_zyx, gt_zyx, pred_zyx, case,
                            out_dir / f"{case}_ct_gt_pred_overlay_best_slice.png", has_gt=has_gt)

    print("Creating overlay video...")
    viz.save_overlay_video(
        image_zyx, gt_zyx, pred_zyx, case,
        mp4_path=out_dir / f"{case}_ct_gt_pred_overlay_video.mp4",
        gif_path=out_dir / f"{case}_ct_gt_pred_overlay_video.gif",
        fps=12, step=2, has_gt=has_gt,
    )

    # ----- explainability artifacts -----
    # Collected here so the 3D HTML (built last) can color the predicted mesh by them.
    html_overlays: list[dict] = []
    if not args.no_explain:
        print("\n--- Explainability ---")
        entropy_sitk = uncertainty["entropy"]
        std_sitk = uncertainty["std"]
        mean_prob_sitk = uncertainty["mean_prob"]

        sitk.WriteImage(entropy_sitk, str(out_dir / f"{case}_uncertainty_entropy.nii.gz"), True)
        sitk.WriteImage(std_sitk, str(out_dir / f"{case}_uncertainty_std.nii.gz"), True)
        sitk.WriteImage(mean_prob_sitk, str(out_dir / f"{case}_uncertainty_mean_prob.nii.gz"), True)
        print(f"Saved uncertainty NIfTIs (mean_prob, entropy, std) in {out_dir}/")

        entropy_zyx = sitk.GetArrayFromImage(entropy_sitk).astype(np.float32)
        ln2 = float(np.log(2.0))
        print(f"  entropy range: [{entropy_zyx.min():.4f}, {entropy_zyx.max():.4f}]  (bound: ln 2 = {ln2:.4f})")
        # Concentration check: mean entropy near boundary vs interior
        from scipy.ndimage import binary_dilation, binary_erosion
        if pred_zyx.any():
            boundary = binary_dilation(pred_zyx, iterations=2) & ~binary_erosion(pred_zyx, iterations=2)
            outside = ~binary_dilation(pred_zyx, iterations=5)
            be = float(entropy_zyx[boundary].mean()) if boundary.any() else 0.0
            oe = float(entropy_zyx[outside].mean()) if outside.any() else 0.0
            print(f"  entropy at boundary={be:.4f}  in distant-bg={oe:.4f}  (boundary > bg is the expected pattern)")

        viz.save_heatmap_best_slice_png(
            image_zyx, entropy_zyx, gt_zyx, pred_zyx, case,
            title="MC Dropout predictive entropy", cmap="magma",
            out_path=out_dir / f"{case}_uncertainty_overlay_best_slice.png",
            vmin=0.0, vmax=ln2, mask_below=0.05,
        )
        viz.save_heatmap_video(
            image_zyx, entropy_zyx, gt_zyx, pred_zyx, case,
            title="MC Dropout entropy", cmap="magma",
            mp4_path=out_dir / f"{case}_uncertainty_overlay_video.mp4",
            gif_path=out_dir / f"{case}_uncertainty_overlay_video.gif",
            fps=12, step=2, vmin=0.0, vmax=ln2, mask_below=0.05,
        )
        html_overlays.append(dict(
            field=entropy_sitk, title="Prediction · entropy", colorscale="Magma",
            cmin=0.0, cmax=ln2, colorbar_title="Entropy (nats)", sample_radius=1,
        ))

        print(f"Computing Seg-Grad-CAM (target layer: {args.gradcam_layer})...")
        with SegGradCAM(model, target_layer=args.gradcam_layer) as cam:
            cam_sitk, info = cam.run(image_path, device)
        sitk.WriteImage(cam_sitk, str(out_dir / f"{case}_seggradcam.nii.gz"), True)
        print(f"Saved Grad-CAM NIfTI ({'full volume' if info['full_volume'] else 'centered ROI ' + str(info['roi_shape'])})")

        cam_zyx = sitk.GetArrayFromImage(cam_sitk).astype(np.float32)
        print(f"  cam range: [{cam_zyx.min():.4f}, {cam_zyx.max():.4f}]")
        if pred_zyx.any():
            in_pred = float(cam_zyx[pred_zyx].mean())
            out_pred = float(cam_zyx[~pred_zyx].mean())
            print(f"  cam in predicted mask={in_pred:.4f}  outside={out_pred:.4f}  (in > out is the expected pattern)")

        viz.save_heatmap_best_slice_png(
            image_zyx, cam_zyx, gt_zyx, pred_zyx, case,
            title="Seg-Grad-CAM saliency", cmap="inferno",
            out_path=out_dir / f"{case}_seggradcam_overlay_best_slice.png",
            vmin=0.0, vmax=1.0, mask_below=0.05,
        )
        viz.save_heatmap_video(
            image_zyx, cam_zyx, gt_zyx, pred_zyx, case,
            title="Seg-Grad-CAM", cmap="inferno",
            mp4_path=out_dir / f"{case}_seggradcam_overlay_video.mp4",
            gif_path=out_dir / f"{case}_seggradcam_overlay_video.gif",
            fps=12, step=2, vmin=0.0, vmax=1.0, mask_below=0.05,
        )
        html_overlays.append(dict(
            field=cam_sitk, title="Prediction · Seg-Grad-CAM", colorscale="Inferno",
            cmin=0.0, cmax=1.0, colorbar_title="Saliency", sample_radius=1,
        ))

    # ----- 3D HTML (last, so the predicted mesh can be colored by the maps above) -----
    print("\nCreating 3D HTML...")
    viz.save_3d_html(
        gt_smoothed, pred_smoothed, case,
        out_dir / f"{case}_gt_pred_3d_comparison.html",
        pred_overlays=html_overlays or None,
    )

    print(f"\nAll outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
