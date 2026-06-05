"""Sliding-window inference and MC Dropout uncertainty.

prediction_to_sitk bridges MONAI's channel-first tensor back to a SimpleITK image
that shares geometry with the source CT. It tolerates both xyz- and zyx-shaped
prediction tensors.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference

from sega.transforms import make_preprocess, make_postprocess


ROI_SIZE = (160, 160, 160)
SW_BATCH_SIZE = 2


def prediction_to_sitk(pred_tensor: torch.Tensor, reference: sitk.Image, *, dtype=np.uint8) -> sitk.Image:
    """Convert a MONAI prediction tensor to a SimpleITK image in the reference geometry.

    Accepts either a (C, X, Y, Z) channel-first tensor or a squeezeable (X, Y, Z) tensor,
    and handles both xyz- and zyx-shaped axes.
    """
    arr = np.squeeze(pred_tensor.detach().cpu().numpy().astype(dtype if np.issubdtype(dtype, np.integer) else np.float32))
    size_xyz = tuple(reference.GetSize())
    if tuple(arr.shape) == size_xyz:
        arr_zyx = np.transpose(arr, (2, 1, 0))
    elif tuple(arr.shape) == size_xyz[::-1]:
        arr_zyx = arr
    else:
        raise ValueError(f"pred shape {arr.shape} does not match reference xyz {size_xyz}")
    if np.issubdtype(dtype, np.integer):
        arr_zyx = arr_zyx.astype(dtype)
        img = sitk.GetImageFromArray(arr_zyx)
        img.CopyInformation(reference)
        return sitk.Cast(img, sitk.sitkUInt8)
    img = sitk.GetImageFromArray(arr_zyx.astype(np.float32))
    img.CopyInformation(reference)
    return img


def _build_loader(image_path: Path, preprocess):
    dataset = Dataset(data=[{"image": str(image_path)}], transform=preprocess)
    return DataLoader(dataset, batch_size=1, num_workers=0)


def run_inference(image_path: Path, model: torch.nn.Module, device: torch.device) -> sitk.Image:
    preprocess = make_preprocess()
    postprocess = make_postprocess(preprocess, argmax=True)
    reference = sitk.ReadImage(str(image_path))

    loader = _build_loader(image_path, preprocess)
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
    return prediction_to_sitk(pred_tensor, reference, dtype=np.uint8)


def _enable_dropout(model: torch.nn.Module) -> int:
    """Put only nn.Dropout* submodules in training mode; leave norm/etc in eval.

    Returns the count of enabled dropout modules so the caller can sanity-check.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            m.train()
            n += 1
    return n


def run_mc_inference(
    image_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    n_passes: int = 20,
) -> dict:
    """Run sliding-window inference n_passes times with dropout active.

    Returns a dict with SimpleITK images in the original CT geometry:
        pred      : argmax mask (from mean foreground probability)
        mean_prob : foreground probability averaged across passes
        entropy   : predictive entropy of the mean foreground probability,
                    bounded in [0, ln 2] for the binary case
        std       : per-voxel std of foreground probability across passes
    """
    if n_passes < 1:
        raise ValueError("n_passes must be >= 1")

    preprocess = make_preprocess()
    postprocess_prob = make_postprocess(preprocess, argmax=False, nearest_interp=False)
    postprocess_mask = make_postprocess(preprocess, argmax=True, nearest_interp=False)
    reference = sitk.ReadImage(str(image_path))

    loader = _build_loader(image_path, preprocess)
    batch = next(iter(loader))
    inputs = batch["image"].to(device)

    model.eval()
    n_dropout = _enable_dropout(model)
    if n_dropout == 0:
        raise RuntimeError("Model has no Dropout submodules; MC Dropout cannot run.")

    fg_prob_sum = None
    fg_prob_sq_sum = None

    with torch.no_grad():
        for _ in range(n_passes):
            logits = sliding_window_inference(
                inputs=inputs,
                roi_size=ROI_SIZE,
                sw_batch_size=SW_BATCH_SIZE,
                predictor=model,
                overlap=0.25,
            )
            probs = torch.softmax(logits, dim=1)
            fg = probs[:, 1:2]  # keep channel dim for postprocess invert
            if fg_prob_sum is None:
                fg_prob_sum = fg.clone()
                fg_prob_sq_sum = fg.pow(2).clone()
            else:
                fg_prob_sum += fg
                fg_prob_sq_sum += fg.pow(2)

    model.eval()  # restore

    mean_prob = fg_prob_sum / n_passes
    var = (fg_prob_sq_sum / n_passes - mean_prob.pow(2)).clamp(min=0.0)
    std = var.sqrt()

    eps = 1e-7
    p = mean_prob.clamp(min=eps, max=1.0 - eps)
    entropy = -(p * p.log() + (1.0 - p) * (1.0 - p).log())  # in [0, ln 2]

    def _invert(field: torch.Tensor, transform, dtype) -> sitk.Image:
        batch["pred"] = field
        restored = [transform(it) for it in decollate_batch(batch)]
        return prediction_to_sitk(restored[0]["pred"], reference, dtype=dtype)

    mean_prob_sitk = _invert(mean_prob, postprocess_prob, dtype=np.float32)
    entropy_sitk = _invert(entropy, postprocess_prob, dtype=np.float32)
    std_sitk = _invert(std, postprocess_prob, dtype=np.float32)

    twoch_mean = torch.cat([1.0 - mean_prob, mean_prob], dim=1)
    pred_sitk = _invert(twoch_mean, postprocess_mask, dtype=np.uint8)

    return {
        "pred": pred_sitk,
        "mean_prob": mean_prob_sitk,
        "entropy": entropy_sitk,
        "std": std_sitk,
    }
