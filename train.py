#!/usr/bin/env python
# coding: utf-8
"""Train / fine-tune the SEG.A aorta SegResNet.

Reproduces the pipeline that produced ``models/checkpoint_segresnet.pth`` (binary
aorta-vs-background segmentation) and additionally supports the AortaSeg25
grand-challenge data layout. The model topology and save format match
``sega.model.build_segresnet`` / ``sega.model.load_checkpoint`` so any checkpoint
written here loads straight back into the inference pipeline (``demo.py``).

Two on-disk layouts are supported (``--dataset-format``):

  sega        data_dir/volumes_full/<CASE>.nii.gz  +  data_dir/labels_full/<CASE>.nii.gz
  aortaseg25  data_dir/images/<subj>_CTA.mha       +  data_dir/masks/<subj>_label.mha

AortaSeg25 masks are multi-label (~23 segments); they are binarized (label > 0 -> 1)
so the model stays out_channels=2 and remains checkpoint-compatible.

Modes:
  train     train from scratch.
  finetune  continue training from --resume (required) weights.

Smoke test (--smoke-test): run a few training steps on whatever local data exists,
report per-sample wall time and peak GPU VRAM. Use --batch-size 4 to size memory.

Examples:
  python train.py --smoke-test --batch-size 4 --data-dir data --device cuda:0
  python train.py --mode train --data-dir data --max-epochs 600 --output-dir runs/sega
  python train.py --mode finetune --resume models/checkpoint_segresnet.pth --data-dir data
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import random
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import torch

from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    Lambdad,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
)
from monai.networks.nets import SegResNet  # noqa: F401  (kept for parity / introspection)
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, decollate_batch

from sega.model import build_segresnet, load_checkpoint

torch.multiprocessing.set_sharing_strategy("file_system")

ROI_SIZE = (160, 160, 160)
SW_BATCH_SIZE = 4
PATCH_SIZE = (96, 96, 96)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train / fine-tune the SEG.A aorta SegResNet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["train", "finetune"], default="train")
    p.add_argument("--dataset-format", choices=["sega", "aortaseg25", "auto"], default="auto")
    p.add_argument("--data-dir", default="data",
                   help="Root containing volumes_full/+labels_full/ (sega) or images/+masks/ (aortaseg25).")
    p.add_argument("--resume", default=None, help="Checkpoint to load (required for --mode finetune).")
    p.add_argument("--output-dir", default=None,
                   help="Where to write best_metric_model.pth (default: runs/<timestamp>).")
    p.add_argument("--device", default=None, help="cuda, cuda:N, or cpu (default: auto).")

    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-samples", type=int, default=4, help="RandCrop patches per volume.")
    p.add_argument("--max-epochs", type=int, default=600)
    p.add_argument("--val-interval", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--cache-rate", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-split", type=float, default=0.2, help="Val fraction when no --folds-csv given.")
    p.add_argument("--seed", type=int, default=0)

    # Original-dataset fold compatibility (optional).
    p.add_argument("--folds-csv", default=None, help="CSV of folds (sega layout); overrides --val-split.")
    p.add_argument("--fold-num", type=int, default=0)

    # Smoke test.
    p.add_argument("--smoke-test", action="store_true",
                   help="Run a few timed training steps and report per-sample time + peak VRAM.")
    p.add_argument("--smoke-steps", type=int, default=3, help="Timed steps in smoke test.")
    p.add_argument("--smoke-warmup", type=int, default=1, help="Un-timed warmup steps in smoke test.")

    args = p.parse_args()
    if args.mode == "finetune" and not args.resume:
        p.error("--mode finetune requires --resume <ckpt>")
    return args


def resolve_device(spec: str | None) -> torch.device:
    device = torch.device(spec) if spec else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    return device


# --------------------------------------------------------------------------- #
# Dataset discovery
# --------------------------------------------------------------------------- #
def detect_format(data_dir: Path) -> str:
    if (data_dir / "volumes_full").is_dir():
        return "sega"
    if (data_dir / "images").is_dir():
        return "aortaseg25"
    raise FileNotFoundError(
        f"Could not auto-detect dataset format under {data_dir}: "
        "expected a 'volumes_full/' (sega) or 'images/' (aortaseg25) subfolder."
    )


def build_file_list(data_dir: Path, fmt: str) -> list[dict]:
    """Discover image/label pairs for the chosen layout. Skips images with no label."""
    if fmt == "sega":
        images = sorted(glob.glob(str(data_dir / "volumes_full" / "*.nii.gz")))
        label_of = lambda img: img.replace("volumes_full", "labels_full")
    elif fmt == "aortaseg25":
        images = sorted(glob.glob(str(data_dir / "images" / "*_CTA.mha")))
        label_of = lambda img: img.replace(f"{os.sep}images{os.sep}", f"{os.sep}masks{os.sep}").replace(
            "_CTA.mha", "_label.mha"
        )
    else:
        raise ValueError(f"Unknown dataset format: {fmt}")

    files = []
    for img in images:
        lab = label_of(img)
        if os.path.exists(lab):
            files.append({"image": img, "label": lab})
        else:
            print(f"  [warn] no label for {img} (expected {lab}); skipping")
    if not files:
        raise FileNotFoundError(f"No image/label pairs found under {data_dir} for format '{fmt}'.")
    return files


def split_files(files: list[dict], args: argparse.Namespace, fmt: str) -> tuple[list[dict], list[dict]]:
    """Train/val split. Uses --folds-csv for the sega layout if provided, else a shuffled ratio split."""
    if args.folds_csv and fmt == "sega" and os.path.exists(args.folds_csv):
        train_imgs, val_imgs = _load_fold(args.folds_csv, args.fold_num)
        as_pair = lambda img: {"image": img, "label": img.replace("volumes_full", "labels_full")}
        return [as_pair(i) for i in train_imgs], [as_pair(i) for i in val_imgs]

    files = list(files)
    random.Random(args.seed).shuffle(files)
    n_val = max(1, int(round(len(files) * args.val_split))) if len(files) > 1 else 0
    val_files = files[:n_val]
    train_files = files[n_val:] or files  # never leave train empty (tiny datasets)
    return train_files, val_files


def _load_fold(path: str, fold_num: int) -> tuple[list[str], list[str]]:
    """Read a folds CSV: each row is one fold; columns split train vs val by a 'val'/'train' tag,
    or fall back to: rows == folds, the fold_num row is val, the rest train.

    The exact original 'create_folds_utils' is not in this repo, so this is a permissive reader:
    it accepts a CSV whose rows hold image paths and a leading fold id, e.g.
        fold,split,image
        0,train,/.../volumes_full/K18.nii.gz
        0,val,/.../volumes_full/K19.nii.gz
    """
    train, val = [], []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        rows = [r for r in reader if r]
    # Skip a header if present.
    if rows and not rows[0][0].lstrip("-").isdigit():
        rows = rows[1:]
    for row in rows:
        if len(row) < 3:
            continue
        fold, split, image = row[0].strip(), row[1].strip().lower(), row[2].strip()
        if int(fold) != fold_num:
            continue
        (val if split.startswith("val") else train).append(image)
    if not train or not val:
        raise ValueError(
            f"Fold {fold_num} in {path} did not yield both train and val image lists. "
            "Expected rows of: fold,split(train|val),image_path"
        )
    return train, val


# --------------------------------------------------------------------------- #
# Transforms (order matches the reference TRAINING code that produced the checkpoint)
# --------------------------------------------------------------------------- #
def _binarize_label() -> Lambdad:
    # Multi-label AortaSeg25 masks -> {0,1}; a no-op on already-binary SEG.A labels.
    return Lambdad(keys=["label"], func=lambda x: (x > 0).to(x.dtype))


def _intensity_transforms() -> list:
    # Double ScaleIntensityRanged is intentional: HU clamp +-3000, then window -275..1900 -> 0..1.
    return [
        ScaleIntensityRanged(keys=["image"], a_min=-3000, a_max=3000, b_min=-3000, b_max=3000, clip=True),
        ScaleIntensityRanged(keys=["image"], a_min=-275, a_max=1900, b_min=0.0, b_max=1.0, clip=True),
    ]


def make_train_transforms(num_samples: int) -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            _binarize_label(),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.5), mode=("bilinear", "nearest")),
            *_intensity_transforms(),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=PATCH_SIZE,
                pos=1,
                neg=1,
                num_samples=num_samples,
                image_key="image",
                image_threshold=0,
            ),
        ]
    )


def make_val_transforms() -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            _binarize_label(),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.5), mode=("bilinear", "nearest")),
            *_intensity_transforms(),
            CropForegroundd(keys=["image", "label"], source_key="image"),
        ]
    )


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #
def run_smoke_test(args: argparse.Namespace, device: torch.device, train_files: list[dict]) -> None:
    bs = args.batch_size
    # With few volumes, pad by repeating so a genuine `bs`-volume batch can form.
    files = list(train_files)
    padded = False
    if len(files) < bs:
        reps = (bs + len(files) - 1) // len(files)
        files = (files * reps)[: max(bs, len(files))]
        padded = True

    print("\n=== SMOKE TEST ===")
    print(f"volumes available: {len(train_files)}  |  batch size: {bs}"
          + (f"  (padded to {len(files)} by repetition)" if padded else ""))

    ds = CacheDataset(data=files, transform=make_train_transforms(args.num_samples),
                      cache_rate=1.0, num_workers=args.num_workers)
    loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=args.num_workers, drop_last=True)

    model = build_segresnet().to(device)
    if args.mode == "finetune":
        model = load_checkpoint(model, Path(args.resume), device)
    model.train()
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    def steps(n: int):
        produced = 0
        while produced < n:
            for batch in loader:
                if produced >= n:
                    break
                yield batch
                produced += 1

    def one_step(batch) -> tuple[float, int, int]:
        inputs = batch["image"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        return loss.item(), inputs.shape[0], 1  # patches in batch, 1 step

    # Warmup (un-timed): builds CUDA caches/kernels so timing is representative.
    for batch in steps(args.smoke_warmup):
        one_step(batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    times, patches_per_step = [], None
    for batch in steps(args.smoke_steps):
        t0 = time.perf_counter()
        loss_val, n_patches, _ = one_step(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0
        times.append(dt)
        patches_per_step = n_patches
        print(f"  step: {dt:.3f}s  patches={n_patches}  loss={loss_val:.4f}")

    mean_step = sum(times) / len(times)
    volumes_per_step = bs  # drop_last=True guarantees full batches

    print("\n--- smoke test summary ---")
    print(f"device                : {device}"
          + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))
    print(f"volume batch size     : {volumes_per_step}")
    print(f"patches / step        : {patches_per_step}  (batch_size x num_samples = {bs} x {args.num_samples})")
    print(f"mean step time        : {mean_step:.3f}s  (over {len(times)} steps)")
    print(f"time / volume         : {mean_step / volumes_per_step:.3f}s")
    print(f"time / patch          : {mean_step / patches_per_step:.4f}s")
    if device.type == "cuda":
        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        peak_resv = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
        print(f"peak VRAM allocated   : {peak_alloc:.2f} GiB")
        print(f"peak VRAM reserved    : {peak_resv:.2f} GiB")
        print("(process-local peaks via torch.cuda; unaffected by other jobs on a shared GPU)")
    print("--------------------------")


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(args: argparse.Namespace, device: torch.device,
          train_files: list[dict], val_files: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "best_metric_model.pth"

    train_ds = CacheDataset(data=train_files, transform=make_train_transforms(args.num_samples),
                            cache_rate=args.cache_rate, num_workers=args.num_workers)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_ds = CacheDataset(data=val_files, transform=make_val_transforms(),
                          cache_rate=args.cache_rate, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=args.num_workers)

    model = build_segresnet().to(device)
    if args.mode == "finetune":
        print(f"Fine-tuning from {args.resume}")
        model = load_checkpoint(model, Path(args.resume), device)  # strict load, then train below

    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    best_metric, best_metric_epoch = -1.0, -1
    for epoch in range(args.max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{args.max_epochs}")
        model.train()
        epoch_loss, step = 0.0, 0
        for batch_data in train_loader:
            step += 1
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"{step}/{max(1, len(train_ds) // args.batch_size)}, train_loss: {loss.item():.4f}")
        epoch_loss /= max(1, step)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if val_files and (epoch + 1) % args.val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs = val_data["image"].to(device)
                    val_labels = val_data["label"].to(device)
                    val_outputs = sliding_window_inference(val_inputs, ROI_SIZE, SW_BATCH_SIZE, model)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    dice_metric(y_pred=val_outputs, y=val_labels)
                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                if metric > best_metric:
                    best_metric, best_metric_epoch = metric, epoch + 1
                    torch.save(model.state_dict(), str(ckpt_path))
                    print(f"saved new best metric model -> {ckpt_path}")
                print(f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                      f"\nbest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}")

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")


# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    set_determinism(seed=args.seed)
    random.seed(args.seed)

    device = resolve_device(args.device)
    data_dir = Path(args.data_dir)
    fmt = detect_format(data_dir) if args.dataset_format == "auto" else args.dataset_format
    print(f"Dataset format: {fmt}  |  data dir: {data_dir}")

    files = build_file_list(data_dir, fmt)
    train_files, val_files = split_files(files, args, fmt)
    print(f"Discovered {len(files)} pairs -> {len(train_files)} train / {len(val_files)} val")

    if args.smoke_test:
        run_smoke_test(args, device, train_files)
        return

    output_dir = Path(args.output_dir) if args.output_dir else Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S")
    train(args, device, train_files, val_files, output_dir)


if __name__ == "__main__":
    main()
