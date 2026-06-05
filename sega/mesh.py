"""Marching cubes + optional pymeshfix repair + Laplacian smoothing.

Arrays from SimpleITK are zyx-shaped; SimpleITK images, sizes, and spacings are
xyz-shaped. vertices_array_to_physical converts marching-cubes verts (in zyx
index space) to xyz physical coordinates using the image's direction matrix --
this is what makes the OBJ meshes align with the source CT.
"""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk
import trimesh
from skimage import measure

try:
    import pymeshfix
    HAS_MESHFIX = True
except Exception as exc:
    HAS_MESHFIX = False
    print(f"pymeshfix not available, repair will be skipped. Reason: {exc}")


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
