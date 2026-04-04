from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from timm.data import Mixup
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TINY_IMAGE_SIZE = 64


def _get(cfg, *keys, default=None):
    cur = cfg
    for key in keys:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def _validate_imagefolder_root(path: Path, split: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{split} directory not found: {path}. Expected Tiny-ImageNet-200 in ImageFolder layout."
        )
    if not path.is_dir():
        raise NotADirectoryError(f"{split} path is not a directory: {path}")
    class_dirs = [p for p in path.iterdir() if p.is_dir()]
    if not class_dirs:
        raise ValueError(
            f"{split} directory has no class subdirectories: {path}. "
            "ImageFolder requires split/class_name/*.jpg layout, and val_annotations.txt is not auto-processed."
        )


def _build_train_transform(cfg):
    img_size = int(_get(cfg, "data", "img_size", default=224))
    crop_padding = int(_get(cfg, "data", "train_crop_padding", default=4))
    randaugment_num_ops = int(_get(cfg, "data", "randaugment_num_ops", default=2))
    randaugment_magnitude = int(_get(cfg, "data", "randaugment_magnitude", default=9))
    random_erasing_prob = float(_get(cfg, "data", "random_erasing_prob", default=0.25))

    ops = [
        transforms.RandomCrop(TINY_IMAGE_SIZE, padding=crop_padding, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(
            (img_size, img_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
    ]
    if randaugment_num_ops > 0 and randaugment_magnitude > 0:
        ops.append(
            transforms.RandAugment(
                num_ops=randaugment_num_ops,
                magnitude=randaugment_magnitude,
                interpolation=InterpolationMode.BICUBIC,
                fill=[int(channel * 255) for channel in IMAGENET_MEAN],
            )
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    if random_erasing_prob > 0:
        ops.append(transforms.RandomErasing(p=random_erasing_prob, value=0))
    return transforms.Compose(ops)


def _build_val_transform(cfg):
    img_size = int(_get(cfg, "data", "img_size", default=224))
    return transforms.Compose(
        [
            transforms.Resize(
                (img_size, img_size),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _build_mixup(cfg):
    model_cfg = cfg["model"]
    mixup_alpha = float(_get(cfg, "train", "mixup_alpha", default=0.0))
    cutmix_alpha = float(_get(cfg, "train", "cutmix_alpha", default=0.0))
    label_smoothing = float(_get(cfg, "train", "label_smoothing", default=0.1))

    if mixup_alpha <= 0.0 and cutmix_alpha <= 0.0:
        return None

    return Mixup(
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        cutmix_minmax=None,
        prob=float(_get(cfg, "train", "mixup_prob", default=1.0)),
        switch_prob=float(_get(cfg, "train", "mixup_switch_prob", default=0.5)),
        mode="batch",
        correct_lam=True,
        label_smoothing=label_smoothing,
        num_classes=int(model_cfg["num_classes"]),
    )


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_loader(cfg):
    root = Path(_get(cfg, "data", "root", default="dataset/tiny-imagenet-200"))
    batch_size = int(_get(cfg, "data", "batch_size", default=64))
    num_workers = int(_get(cfg, "data", "num_workers", default=4))
    seed = int(_get(cfg, "train", "seed", default=42))

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    val_generator = torch.Generator()
    val_generator.manual_seed(seed + 1)

    train_root = root / "train"
    val_root = root / "val"
    _validate_imagefolder_root(train_root, "train")
    _validate_imagefolder_root(val_root, "val")

    train_transform = _build_train_transform(cfg)
    val_transform = _build_val_transform(cfg)

    try:
        train_dataset = datasets.ImageFolder(train_root, transform=train_transform)
    except Exception as exc:
        raise RuntimeError(f"Failed to build train ImageFolder from {train_root}: {exc}") from exc

    try:
        val_dataset = datasets.ImageFolder(val_root, transform=val_transform)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to build val ImageFolder from {val_root}: {exc}. "
            "This loader does not convert Tiny-ImageNet val_annotations.txt."
        ) from exc

    mixup_fn = _build_mixup(cfg)
    train_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=mixup_fn is not None,
        worker_init_fn=_seed_worker,
        generator=train_generator,
    )
    if num_workers > 0:
        train_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        **train_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_seed_worker,
        generator=val_generator,
    )
    return train_loader, val_loader, mixup_fn
