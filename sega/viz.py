"""2D/3D visualization helpers.

Label overlays use distinct colors per pixel category (GT/Pred/intersection).
Heatmap overlays use a continuous matplotlib colormap, optionally masked to the
predicted foreground or thresholded.
"""
from __future__ import annotations

import math
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import SimpleITK as sitk
import trimesh
from matplotlib.patches import Patch
from plotly.subplots import make_subplots


# ----- shared helpers -----

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


def dice_score(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return 2.0 * np.logical_and(a, b).sum() / denom


def _slice_range_around_labels(gt: np.ndarray, pred: np.ndarray, step: int, margin: int) -> list[int]:
    combined = gt | pred
    counts = combined.sum(axis=(1, 2))
    nonzero = np.flatnonzero(counts)
    if len(nonzero) == 0:
        return list(range(0, gt.shape[0], step))
    start = max(0, int(nonzero[0]) - margin)
    stop = min(gt.shape[0], int(nonzero[-1]) + margin + 1)
    return list(range(start, stop, step))


def _save_frames(frames: list[np.ndarray], mp4_path: Path, gif_path: Path, fps: int) -> Path:
    try:
        imageio.mimsave(mp4_path, frames, fps=fps, codec="libx264", macro_block_size=1)
        print(f"Saved MP4: {mp4_path}")
        return mp4_path
    except Exception as exc:
        print(f"MP4 failed: {exc}; writing GIF instead")
        imageio.mimsave(gif_path, frames, duration=1.0 / fps)
        print(f"Saved GIF: {gif_path}")
        return gif_path


# ----- label (GT/Pred) overlay -----

def make_label_overlay_rgba(gt: np.ndarray, pred: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    rgba = np.zeros((*gt.shape, 4), dtype=np.float32)
    rgba[gt & ~pred] = (0.0, 1.0, 0.0, alpha)
    rgba[pred & ~gt] = (1.0, 0.0, 1.0, alpha)
    rgba[gt & pred] = (1.0, 1.0, 0.0, alpha)
    return rgba


def save_best_slice_png(image: np.ndarray, gt: np.ndarray, pred: np.ndarray, case_id: str, out_path: Path,
                        has_gt: bool = True):
    z = choose_best_axial_slice(gt, pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(normalize_for_display(image[z]), cmap="gray")
    ax.imshow(make_label_overlay_rgba(gt[z], pred[z]))
    if has_gt:
        ax.set_title(f"{case_id}: CT + GT + Pred | z={z} | Dice={dice_score(gt, pred):.4f}", fontsize=14)
        handles = [
            Patch(facecolor=(0.0, 1.0, 0.0, 0.55), label="GT only"),
            Patch(facecolor=(1.0, 0.0, 1.0, 0.55), label="Prediction only"),
            Patch(facecolor=(1.0, 1.0, 0.0, 0.55), label="GT ∩ Pred"),
        ]
    else:
        ax.set_title(f"{case_id}: CT + Pred | z={z}", fontsize=14)
        handles = [Patch(facecolor=(1.0, 0.0, 1.0, 0.55), label="Prediction")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved overlay PNG: {out_path}")


def _render_overlay_frame(image, gt, pred, z, case_id, overall_dice, dpi=120):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    ax.imshow(normalize_for_display(image[z]), cmap="gray")
    ax.imshow(make_label_overlay_rgba(gt[z], pred[z]))
    title = f"{case_id} | axial z={z}"
    if overall_dice is not None:
        title += f" | Dice={overall_dice:.4f}"
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout(pad=0.2)
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def save_overlay_video(image, gt, pred, case_id, mp4_path: Path, gif_path: Path, fps=12, step=2, margin=5,
                       has_gt=True):
    slices = _slice_range_around_labels(gt, pred, step, margin)
    print(f"Rendering {len(slices)} GT/Pred video frames...")
    overall_dice = dice_score(gt, pred) if has_gt else None
    frames = [_render_overlay_frame(image, gt, pred, z, case_id, overall_dice) for z in slices]
    return _save_frames(frames, mp4_path, gif_path, fps)


# ----- heatmap (continuous map) overlay -----

def make_heatmap_overlay_rgba(
    heatmap_2d: np.ndarray,
    cmap: str = "magma",
    alpha: float = 0.55,
    vmin: float | None = None,
    vmax: float | None = None,
    mask_below: float | None = None,
) -> np.ndarray:
    """RGBA overlay from a continuous heatmap.

    mask_below: voxels with heatmap value below this threshold are made fully transparent
    so the underlying CT shows through (use to hide near-zero background activations).
    """
    if vmin is None:
        vmin = float(np.nanmin(heatmap_2d))
    if vmax is None:
        vmax = float(np.nanmax(heatmap_2d))
    if vmax <= vmin:
        vmax = vmin + 1e-8
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    rgba = matplotlib.colormaps[cmap](norm(heatmap_2d))
    rgba[..., 3] = alpha
    if mask_below is not None:
        rgba[heatmap_2d < mask_below, 3] = 0.0
    return rgba.astype(np.float32)


def save_heatmap_best_slice_png(
    image: np.ndarray,
    heatmap: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    case_id: str,
    title: str,
    cmap: str,
    out_path: Path,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    mask_below: float | None = None,
    contour_pred: bool = True,
):
    """Save a single best-slice PNG showing CT + heatmap, with predicted-mask contour."""
    z = choose_best_axial_slice(gt, pred)
    vmin_eff = float(np.nanmin(heatmap)) if vmin is None else vmin
    vmax_eff = float(np.nanmax(heatmap)) if vmax is None else vmax
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(normalize_for_display(image[z]), cmap="gray")
    ax.imshow(
        make_heatmap_overlay_rgba(
            heatmap[z], cmap=cmap, alpha=0.55, vmin=vmin_eff, vmax=vmax_eff, mask_below=mask_below,
        )
    )
    if contour_pred and pred[z].any():
        ax.contour(pred[z].astype(np.uint8), levels=[0.5], colors="cyan", linewidths=0.8)
    ax.set_title(f"{case_id}: {title} | z={z}", fontsize=13)
    ax.axis("off")
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(vmin=vmin_eff, vmax=vmax_eff))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap PNG: {out_path}")


def _render_heatmap_frame(
    image, heatmap, pred, z, case_id, title, cmap, vmin, vmax, mask_below, dpi=110,
):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    ax.imshow(normalize_for_display(image[z]), cmap="gray")
    ax.imshow(
        make_heatmap_overlay_rgba(
            heatmap[z], cmap=cmap, alpha=0.55, vmin=vmin, vmax=vmax, mask_below=mask_below,
        )
    )
    if pred[z].any():
        ax.contour(pred[z].astype(np.uint8), levels=[0.5], colors="cyan", linewidths=0.6)
    ax.set_title(f"{case_id} | {title} | z={z}")
    ax.axis("off")
    plt.tight_layout(pad=0.2)
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def save_heatmap_video(
    image: np.ndarray,
    heatmap: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    case_id: str,
    title: str,
    cmap: str,
    mp4_path: Path,
    gif_path: Path,
    *,
    fps: int = 12,
    step: int = 2,
    margin: int = 5,
    vmin: float | None = None,
    vmax: float | None = None,
    mask_below: float | None = None,
):
    slices = _slice_range_around_labels(gt, pred, step, margin)
    vmin_eff = float(np.nanmin(heatmap)) if vmin is None else vmin
    vmax_eff = float(np.nanmax(heatmap)) if vmax is None else vmax
    print(f"Rendering {len(slices)} {title} frames...")
    frames = [
        _render_heatmap_frame(
            image, heatmap, pred, z, case_id, title, cmap, vmin_eff, vmax_eff, mask_below,
        )
        for z in slices
    ]
    return _save_frames(frames, mp4_path, gif_path, fps)


# ----- 3D mesh HTML -----

def mesh_to_trace(mesh: trimesh.Trimesh, name: str, color: str = "red") -> go.Mesh3d:
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.faces)
    return go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        name=name, color=color, opacity=1.0, flatshading=False,
        lighting=dict(ambient=0.25, diffuse=0.75, specular=0.35, roughness=0.55, fresnel=0.2),
        lightposition=dict(x=100, y=200, z=300),
        showscale=False,
    )


def sample_field_at_vertices(field: sitk.Image, vertices_xyz: np.ndarray, radius: int = 0) -> np.ndarray:
    """Sample a scalar SITK image at mesh vertex physical (xyz) coordinates.

    Mesh vertices live in the source CT's physical space and the field image
    (entropy / Grad-CAM / std) shares that geometry, so each vertex maps to a
    voxel index via the inverse of (origin + direction @ (index * spacing)).
    Out-of-bounds vertices clamp to the edge. Returns a (N,) float32 array.

    ``radius``: 0 -> nearest-neighbour read. ``radius >= 1`` returns the **max**
    over the (2r+1)^3 voxel neighborhood around each vertex. The mesh sits exactly
    on the segmentation boundary, where fields like MC-Dropout entropy live in a
    razor-thin shell; nearest-neighbour misses it and renders black, so a small
    neighborhood-max lets that shell register on the surface.
    """
    arr = sitk.GetArrayFromImage(field).astype(np.float32)             # zyx
    size = np.array(field.GetSize(), dtype=np.int64)                   # xyz
    origin = np.array(field.GetOrigin(), dtype=np.float64)             # xyz
    spacing = np.array(field.GetSpacing(), dtype=np.float64)           # xyz
    direction = np.array(field.GetDirection(), dtype=np.float64).reshape(3, 3)
    rel = np.asarray(vertices_xyz, dtype=np.float64) - origin
    cont_idx = (rel @ np.linalg.inv(direction).T) / spacing            # (N,3) xyz
    base = np.rint(cont_idx).astype(np.int64)
    if radius <= 0:
        idx = np.clip(base, 0, size - 1)
        return arr[idx[:, 2], idx[:, 1], idx[:, 0]]
    rng = range(-radius, radius + 1)
    out = None
    for dx in rng:
        for dy in rng:
            for dz in rng:
                idx = np.clip(base + np.array([dx, dy, dz]), 0, size - 1)
                vals = arr[idx[:, 2], idx[:, 1], idx[:, 0]]
                out = vals if out is None else np.maximum(out, vals)
    return out


def mesh_to_intensity_trace(
    mesh: trimesh.Trimesh,
    name: str,
    intensity: np.ndarray,
    colorscale: str,
    cmin: float,
    cmax: float,
    colorbar: dict,
) -> go.Mesh3d:
    """A Mesh3d whose surface is colored per-vertex by a scalar field."""
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.faces)
    return go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        name=name, intensity=np.asarray(intensity, dtype=np.float32), intensitymode="vertex",
        colorscale=colorscale, cmin=cmin, cmax=cmax,
        showscale=True, colorbar=colorbar, opacity=1.0, flatshading=False,
        lighting=dict(ambient=0.6, diffuse=0.55, specular=0.08, roughness=0.9, fresnel=0.1),
        lightposition=dict(x=100, y=200, z=300),
    )


def _scene_blank(fig, row, col):
    fig.update_scenes(
        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
        aspectmode="data", camera=dict(eye=dict(x=1.4, y=-2.0, z=0.9)),
        row=row, col=col,
    )


def save_3d_html(
    gt_mesh: trimesh.Trimesh | None,
    pred_mesh: trimesh.Trimesh,
    case_id: str,
    out_path: Path,
    pred_overlays: list[dict] | None = None,
):
    """Write an interactive side-by-side 3D HTML.

    Without ``pred_overlays`` this is the original GT-vs-PRED two-panel view
    (or a single PRED panel when ``gt_mesh`` is None, i.e. label-free mode).

    With ``pred_overlays`` (a list of dicts, each ``{"field": sitk.Image,
    "title": str, "colorscale": str, "cmin": float, "cmax": float,
    "colorbar_title": str, "sample_radius": int}``) the predicted-mesh surface is
    additionally rendered once per overlay, colored per-vertex by that volumetric
    field (e.g. MC Dropout entropy, Seg-Grad-CAM saliency). ``sample_radius``
    (default 0) is forwarded to ``sample_field_at_vertices`` — use 1 for thin-shell
    fields like entropy. Panels are laid out in a 2-column grid: GT, PRED, then one
    colored prediction per overlay.
    """
    if not pred_overlays:
        if gt_mesh is None:
            fig = make_subplots(
                rows=1, cols=1, specs=[[{"type": "scene"}]],
                subplot_titles=("PRED",),
            )
            fig.add_trace(mesh_to_trace(pred_mesh, "PRED"), row=1, col=1)
            _scene_blank(fig, 1, 1)
            fig.update_layout(
                title=f"{case_id}: prediction 3D aorta mesh",
                width=700, height=900, showlegend=False,
                margin=dict(l=0, r=0, t=80, b=0),
                paper_bgcolor="white", plot_bgcolor="white", font=dict(size=18),
            )
            fig.write_html(str(out_path), include_plotlyjs="cdn")
            print(f"Saved 3D HTML: {out_path}")
            return
        fig = make_subplots(
            rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
            subplot_titles=("GT", "PRED"), horizontal_spacing=0.02,
        )
        fig.add_trace(mesh_to_trace(gt_mesh, "GT"), row=1, col=1)
        fig.add_trace(mesh_to_trace(pred_mesh, "PRED"), row=1, col=2)
        _scene_blank(fig, 1, 1)
        _scene_blank(fig, 1, 2)
        fig.update_layout(
            title=f"{case_id}: ground truth vs prediction 3D aorta mesh",
            width=1000, height=900, showlegend=False,
            margin=dict(l=0, r=0, t=80, b=0),
            paper_bgcolor="white", plot_bgcolor="white", font=dict(size=18),
        )
        fig.write_html(str(out_path), include_plotlyjs="cdn")
        print(f"Saved 3D HTML: {out_path}")
        return

    cols = 2
    panels = [("Prediction", None)] if gt_mesh is None else [("Ground truth", None), ("Prediction", None)]
    for ov in pred_overlays:
        panels.append((ov["title"], ov))
    n = len(panels)
    rows = math.ceil(n / cols)

    hspace, vspace = 0.06, 0.10
    col_w = (1.0 - hspace * (cols - 1)) / cols
    row_h = (1.0 - vspace * (rows - 1)) / rows

    titles = [p[0] for p in panels] + [""] * (rows * cols - n)
    fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{"type": "scene"} for _ in range(cols)] for _ in range(rows)],
        subplot_titles=titles, horizontal_spacing=hspace, vertical_spacing=vspace,
    )

    pred_verts = np.asarray(pred_mesh.vertices)
    for k, (title, ov) in enumerate(panels):
        r, c = k // cols + 1, k % cols + 1
        if ov is None:
            mesh = gt_mesh if (gt_mesh is not None and k == 0) else pred_mesh
            fig.add_trace(mesh_to_trace(mesh, title, color="red"), row=r, col=c)
        else:
            intensity = sample_field_at_vertices(ov["field"], pred_verts, radius=ov.get("sample_radius", 0))
            # colorbar to the right of this panel's column, centered on its row
            x_right = (c - 1) * (col_w + hspace) + col_w
            y_top = 1.0 - (r - 1) * (row_h + vspace)
            colorbar = dict(
                title=dict(text=ov.get("colorbar_title", title), side="right"),
                x=min(x_right + 0.01, 1.0), xanchor="left",
                y=y_top - row_h / 2.0, yanchor="middle",
                len=row_h * 0.9, thickness=14,
            )
            fig.add_trace(
                mesh_to_intensity_trace(
                    pred_mesh, title, intensity,
                    colorscale=ov.get("colorscale", "Magma"),
                    cmin=ov.get("cmin", float(np.nanmin(intensity))),
                    cmax=ov.get("cmax", float(np.nanmax(intensity))),
                    colorbar=colorbar,
                ),
                row=r, col=c,
            )
        _scene_blank(fig, r, c)

    fig.update_layout(
        title=f"{case_id}: 3D aorta — GT vs prediction, with uncertainty & saliency",
        width=1100, height=520 * rows, showlegend=False,
        margin=dict(l=0, r=60, t=80, b=0),
        paper_bgcolor="white", plot_bgcolor="white", font=dict(size=16),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"Saved 3D HTML (with {len(pred_overlays)} explainability overlay(s)): {out_path}")
