"""Post-hoc explainability: Seg-Grad-CAM saliency and MC Dropout uncertainty.

SegGradCAM is a from-scratch implementation rather than a thin wrapper around
monai.visualize.GradCAM because MONAI's GradCAM was designed for classification:
it picks logits[:, class_idx] at a single spatial location (or averages over
all of them) before backprop, which is inappropriate for dense segmentation.
The canonical Seg-Grad-CAM (Vinogradova et al., 2020) instead sums the
foreground logit over the predicted foreground mask, so the saliency answers
"what features made *this* segmentation fire." (Summing rather than averaging
keeps the back-propagated gradient at a usable magnitude; the final CAM is
max-normalized to [0, 1] so the choice does not change the relative map.)

Target layer (default model.up_layers[-1]) is the last decoder ResBlock at full
input resolution, just before the 1x1x1 conv_final. SegResNet's conv_final has
only 2 output channels (bg/fg) -- too few for channel-weighted CAM. The
deepest encoder activations are too coarse spatially.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.data import decollate_batch

from sega.inference import (
    ROI_SIZE,
    _build_loader,
    _enable_dropout,
    prediction_to_sitk,
    run_mc_inference,
)
from sega.transforms import make_preprocess, make_postprocess


def resolve_layer(model: nn.Module, path: str) -> nn.Module:
    """Resolve dotted attribute paths like 'up_layers.-1' on a model.

    Supports integer indices (positive or negative) as path components, used
    when the attribute is a ModuleList/Sequential.
    """
    target = model
    for piece in path.split("."):
        if piece.lstrip("-").isdigit():
            target = target[int(piece)]
        else:
            target = getattr(target, piece)
    return target


class SegGradCAM:
    """Seg-Grad-CAM (Vinogradova et al. 2020) for a 3D segmentation model.

    Usage:
        cam = SegGradCAM(model, target_layer=model.up_layers[-1])
        heatmap_sitk = cam.run(image_path, device, reference=sitk.ReadImage(...))

    The forward used for Grad-CAM is a single full-volume forward on the
    preprocessed input. On CUDA OOM we fall back to a centered 160^3 ROI patch
    (the inference window size) and zero-pad the CAM outside it.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module | str = "up_layers.-1"):
        self.model = model
        if isinstance(target_layer, str):
            target_layer = resolve_layer(model, target_layer)
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._fwd_handle = target_layer.register_forward_hook(self._fwd_hook)
        self._bwd_handle = target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inputs, output):
        self._activations = output

    def _bwd_hook(self, module, grad_input, grad_output):
        self._gradients = grad_output[0]

    def close(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ----- core compute -----

    @staticmethod
    def _pad_to_multiple(t: torch.Tensor, multiple: int = 8) -> tuple[torch.Tensor, tuple[int, ...]]:
        """Right-pad spatial dims to the next multiple of `multiple`. Returns (padded, pad_amounts)."""
        shape = t.shape[2:]
        pads_xyz = tuple((multiple - (s % multiple)) % multiple for s in shape)
        if all(p == 0 for p in pads_xyz):
            return t, pads_xyz
        # F.pad expects (last_dim_lo, last_dim_hi, ..., first_dim_lo, first_dim_hi)
        pad_arg: list[int] = []
        for p in reversed(pads_xyz):
            pad_arg.extend([0, p])
        return F.pad(t, pad_arg, mode="constant", value=0.0), pads_xyz

    def _compute_cam(self, input_tensor: torch.Tensor, pred_mask: torch.Tensor, fg_class: int) -> torch.Tensor:
        """Run forward+backward and return the CAM at input_tensor's spatial shape.

        input_tensor: (1, 1, X, Y, Z) preprocessed CT.
        pred_mask: (1, 1, X, Y, Z) binary mask in the same coordinate system.
        """
        self.model.zero_grad(set_to_none=True)
        self._activations = None
        self._gradients = None

        orig_shape = input_tensor.shape[2:]
        padded_input, pads = self._pad_to_multiple(input_tensor, multiple=8)
        padded_mask, _ = self._pad_to_multiple(pred_mask, multiple=8)

        with torch.enable_grad():
            x = padded_input.detach().clone().requires_grad_(True)
            logits = self.model(x)                        # (1, C, Xp, Yp, Zp)
            fg_logits = logits[:, fg_class:fg_class + 1]  # (1, 1, Xp, Yp, Zp)

            if padded_mask.sum() == 0:
                scalar = fg_logits.mean()
            else:
                # Sum (not mean) over the mask: averaging divides the gradient by
                # the ~thousands of mask voxels, shrinking it to ~1e-11 and making
                # the CAM collapse to zero. The map is max-normalized below, so the
                # sum vs mean choice only affects scale, not the relative pattern.
                scalar = (fg_logits * padded_mask).sum()

            scalar.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Target-layer activations/gradients were not captured. Wrong layer?")

        A = self._activations
        G = self._gradients
        weights = G.mean(dim=(2, 3, 4), keepdim=True)        # (1, K, 1, 1, 1)
        cam = F.relu((weights * A).sum(dim=1, keepdim=True))  # (1, 1, x, y, z)

        # Upsample CAM back to padded input shape, then crop pads off.
        cam = F.interpolate(cam, size=padded_input.shape[2:], mode="trilinear", align_corners=False)
        if any(p > 0 for p in pads):
            sx, sy, sz = orig_shape
            cam = cam[:, :, :sx, :sy, :sz]

        # Max-normalize to [0, 1]. After ReLU the minimum is already 0 (that is
        # the "no relevance" baseline we want to keep), so we divide by the max
        # rather than min-max rescaling. A relative guard on the max avoids the
        # old absolute 1e-8 threshold, which zeroed valid but small-magnitude CAMs.
        c_max = cam.amax()
        if c_max > 0:
            cam = cam / c_max
        else:
            cam = torch.zeros_like(cam)
        return cam.detach()

    def _centered_roi(self, input_tensor: torch.Tensor, pred_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[slice, ...]]:
        """Crop input + mask to a centered ROI_SIZE patch. Returns (input_roi, mask_roi, slices)."""
        shape = input_tensor.shape[2:]
        if pred_mask.sum() > 0:
            # Center on the predicted foreground mass.
            nz = torch.nonzero(pred_mask[0, 0])
            center = nz.float().mean(dim=0).long().tolist()
        else:
            center = [s // 2 for s in shape]
        slices = []
        for c, s, r in zip(center, shape, ROI_SIZE):
            half = r // 2
            lo = max(0, min(s - r, c - half))
            slices.append(slice(lo, lo + min(r, s)))
        slc = (slice(None), slice(None), *slices)
        return input_tensor[slc], pred_mask[slc], slices

    # ----- public entrypoint -----

    def run(
        self,
        image_path: Path,
        device: torch.device,
        *,
        fg_class: int = 1,
        try_full_volume: bool = True,
    ) -> tuple[sitk.Image, dict]:
        """Compute the Seg-Grad-CAM saliency for one CT and return it in the original CT geometry.

        Returns (cam_sitk, info_dict). info_dict carries diagnostic flags:
            full_volume   : True if the full preprocessed volume was used, False if ROI fallback
            roi_shape     : the shape actually used for the Grad-CAM forward
            preproc_shape : the full preprocessed volume's spatial shape
        """
        preprocess = make_preprocess()
        postprocess = make_postprocess(preprocess, argmax=False, nearest_interp=False)
        reference = sitk.ReadImage(str(image_path))

        loader = _build_loader(image_path, preprocess)
        batch = next(iter(loader))
        inputs = batch["image"].to(device)

        # Deterministic pred mask on the same preprocessed volume.
        # Pad to multiple of 8 to satisfy SegResNet's 3-level downsample/upsample symmetry.
        self.model.eval()
        with torch.no_grad():
            padded_inputs, pads = self._pad_to_multiple(inputs, multiple=8)
            pred_logits = self.model(padded_inputs)
            pred_mask_padded = (pred_logits.argmax(dim=1, keepdim=True) == fg_class).float()
            if any(p > 0 for p in pads):
                sx, sy, sz = inputs.shape[2:]
                pred_mask = pred_mask_padded[:, :, :sx, :sy, :sz]
            else:
                pred_mask = pred_mask_padded

        info = {"full_volume": True, "roi_shape": tuple(inputs.shape[2:]), "preproc_shape": tuple(inputs.shape[2:])}

        cam = None
        if try_full_volume:
            try:
                cam = self._compute_cam(inputs, pred_mask, fg_class=fg_class)
            except torch.cuda.OutOfMemoryError:
                print("SegGradCAM: CUDA OOM on full-volume forward; falling back to centered ROI.")
                torch.cuda.empty_cache()
                cam = None

        if cam is None:
            roi_in, roi_mask, slc = self._centered_roi(inputs, pred_mask)
            info["full_volume"] = False
            info["roi_shape"] = tuple(roi_in.shape[2:])
            cam_roi = self._compute_cam(roi_in, roi_mask, fg_class=fg_class)
            cam = torch.zeros_like(inputs)
            cam[(slice(None), slice(None), *slc)] = cam_roi

        # Push the CAM through Invertd so it lands in the original CT geometry.
        batch["pred"] = cam.detach()
        restored = [postprocess(it) for it in decollate_batch(batch)]
        cam_sitk = prediction_to_sitk(restored[0]["pred"], reference, dtype=np.float32)

        # Clamp into [0, 1] after trilinear inversion (may go slightly out of range).
        arr = sitk.GetArrayFromImage(cam_sitk).clip(0.0, 1.0).astype(np.float32)
        out = sitk.GetImageFromArray(arr)
        out.CopyInformation(reference)
        return out, info


# ----- convenience wrapper -----

def mc_dropout_uncertainty(
    model: nn.Module,
    image_path: Path,
    device: torch.device,
    n_passes: int = 20,
) -> dict:
    """Thin pass-through to inference.run_mc_inference; lives here for discoverability."""
    return run_mc_inference(image_path, model, device, n_passes=n_passes)
