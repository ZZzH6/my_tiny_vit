from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from timm.data import Mixup
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


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
            f"{split} directory not found: {path}. Expected Tiny-ImageNet in ImageFolder layout."
        )
    if not path.is_dir():
        raise NotADirectoryError(f"{split} path is not a directory: {path}")
    if not any(child.is_dir() for child in path.iterdir()):
        raise ValueError(f"{split} directory has no class subdirectories: {path}")


def _validate_test_root(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"test directory not found: {path}")
    image_dir = path / "images"
    if not image_dir.exists():
        raise FileNotFoundError(f"test images directory not found: {image_dir}")
    if not any(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in image_dir.iterdir()):
        raise ValueError(f"test images directory has no readable images: {image_dir}")


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _build_train_transform(cfg):
    img_size = int(_get(cfg, "data", "img_size", default=224))
    crop_scale = (
        float(_get(cfg, "data", "train_crop_scale_min", default=0.8)),
        float(_get(cfg, "data", "train_crop_scale_max", default=1.0)),
    )
    crop_ratio = (
        float(_get(cfg, "data", "train_crop_ratio_min", default=0.75)),
        float(_get(cfg, "data", "train_crop_ratio_max", default=1.3333333333333333)),
    )
    randaugment_num_ops = int(_get(cfg, "data", "randaugment_num_ops", default=0))
    randaugment_magnitude = int(_get(cfg, "data", "randaugment_magnitude", default=0))
    random_erasing_prob = float(_get(cfg, "data", "random_erasing_prob", default=0.0))

    ops = [
        transforms.RandomResizedCrop(
            img_size,
            scale=crop_scale,
            ratio=crop_ratio,
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.RandomHorizontalFlip(),
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
        ops.append(transforms.RandomErasing(p=random_erasing_prob, value="random"))
    return transforms.Compose(ops)


def _build_eval_transform(cfg):
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


class TinyImageNetTestDataset(Dataset):
    def __init__(self, image_dir: Path, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.samples = sorted(
            [path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
        )
        if not self.samples:
            raise ValueError(f"No test images found in {image_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path = self.samples[index]
        image = default_loader(str(image_path))
        if self.transform is not None:
            image = self.transform(image)
        return image, image_path.name


def _build_loader_kwargs(batch_size: int, num_workers: int, seed: int, shuffle: bool):
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": True,
        "worker_init_fn": _seed_worker,
        "generator": generator,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    return loader_kwargs


def _build_mixup(cfg):
    mixup_alpha = float(_get(cfg, "train", "mixup_alpha", default=0.0))
    cutmix_alpha = float(_get(cfg, "train", "cutmix_alpha", default=0.0))
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
        label_smoothing=float(_get(cfg, "train", "label_smoothing", default=0.0)),
        num_classes=int(_get(cfg, "model", "num_classes", default=1000)),
    )


def build_loader(cfg):
    root = Path(_get(cfg, "data", "root", default="dataset/tiny-imagenet-200"))
    batch_size = int(_get(cfg, "data", "batch_size", default=64))
    num_workers = int(_get(cfg, "data", "num_workers", default=4))
    seed = int(_get(cfg, "train", "seed", default=42))

    train_root = root / "train"
    val_root = root / "val"
    _validate_imagefolder_root(train_root, "train")
    _validate_imagefolder_root(val_root, "val")

    train_dataset = datasets.ImageFolder(train_root, transform=_build_train_transform(cfg))
    val_dataset = datasets.ImageFolder(val_root, transform=_build_eval_transform(cfg))
    mixup_fn = _build_mixup(cfg)

    train_loader = DataLoader(
        train_dataset,
        drop_last=mixup_fn is not None,
        **_build_loader_kwargs(batch_size, num_workers, seed, shuffle=True),
    )
    val_loader = DataLoader(
        val_dataset,
        **_build_loader_kwargs(batch_size, num_workers, seed + 1, shuffle=False),
    )
    return train_loader, val_loader, mixup_fn


def build_eval_loader(cfg, split: str = "val"):
    root = Path(_get(cfg, "data", "root", default="dataset/tiny-imagenet-200"))
    batch_size = int(_get(cfg, "data", "batch_size", default=64))
    num_workers = int(_get(cfg, "data", "num_workers", default=4))
    seed = int(_get(cfg, "train", "seed", default=42))

    if split == "test":
        test_root = root / "test"
        _validate_test_root(test_root)
        dataset = TinyImageNetTestDataset(test_root / "images", transform=_build_eval_transform(cfg))
        loader = DataLoader(
            dataset,
            **_build_loader_kwargs(batch_size, num_workers, seed + 2, shuffle=False),
        )
        return loader, dataset

    if split not in {"train", "val"}:
        raise ValueError(f"Unsupported split: {split}")

    split_root = root / split
    _validate_imagefolder_root(split_root, split)
    dataset = datasets.ImageFolder(split_root, transform=_build_eval_transform(cfg))
    loader = DataLoader(
        dataset,
        **_build_loader_kwargs(batch_size, num_workers, seed + (0 if split == "train" else 1), shuffle=False),
    )
    return loader, dataset


def get_class_names(cfg) -> list[str]:
    root = Path(_get(cfg, "data", "root", default="dataset/tiny-imagenet-200"))
    train_root = root / "train"
    _validate_imagefolder_root(train_root, "train")
    return sorted(path.name for path in train_root.iterdir() if path.is_dir())
