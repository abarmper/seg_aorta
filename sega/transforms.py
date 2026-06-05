"""Preprocess + postprocess MONAI transforms.

The double ScaleIntensityRanged is intentional: clamp HU to +-3000 first, then
window -275..1900 -> 0..1. Don't merge them.
"""
from __future__ import annotations

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


def make_postprocess(preprocess: Compose, *, argmax: bool = True, nearest_interp: bool = False) -> Compose:
    """Invert preprocess transforms so predictions land back in the original image geometry.

    argmax=False is useful when the caller wants a continuous probability or saliency
    volume (e.g. MC Dropout mean prob, Seg-Grad-CAM) rather than a discrete mask.
    """
    steps = [
        Invertd(
            keys="pred",
            transform=preprocess,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=nearest_interp,
            to_tensor=True,
        ),
    ]
    if argmax:
        steps.append(AsDiscreted(keys="pred", argmax=True))
    return Compose(steps)
