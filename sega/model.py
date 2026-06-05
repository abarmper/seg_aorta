"""SegResNet construction and checkpoint loading.

The topology here must match the checkpoint exactly: changing init_filters,
blocks_down, blocks_up, or upsample_mode breaks load_state_dict(strict=True).
"""
from __future__ import annotations

from pathlib import Path

import torch

from monai.networks.nets import SegResNet
from monai.utils.enums import UpsampleMode


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
