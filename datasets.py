"""
datasets.py - 多数据集注册表与智能数据加载器
==============================================
通过 Dataset Registry 统一管理不同数据集的配置信息。

核心设计原则：
  - 使用数据集的 **原生分辨率**，不进行上采样（Resize / Padding）
  - 通过模型的 Conv Stem (等效 patch_size=4) 自适应不同分辨率
  - 对于远大于模型需求的图像，使用 RandomResizedCrop 降采样（标准学术做法）
"""

import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# =====================================================================
#                       Dataset Registry (数据集注册表)
# =====================================================================
# 每个数据集配置包含:
#   - class_fn:      torchvision 数据集类 (callable)
#   - num_classes:    分类类别数
#   - native_size:    原始图像分辨率
#   - img_size:       送入模型的目标分辨率 (= native_size 或合理缩小)
#   - mean / std:     通道归一化统计量
#   - auto_augment:   AutoAugment 策略名 (str or None)
#
# 分辨率适配原则:
#   1. 小图 (32x32, 64x64): 直接使用原生分辨率 + RandomCrop(padding=4)
#      不做任何上采样，模型 Conv Stem 自然生成对应大小的 Token 序列
#   2. 大图 (96x96+):       使用 RandomResizedCrop 缩至 64x64
# =====================================================================

DATASET_REGISTRY = {
    'cifar100': {
        'class_fn': torchvision.datasets.CIFAR100,
        'num_classes': 100,
        'native_size': 32,
        'img_size': 32,       # ← 使用原生 32x32，不上采样
        'mean': [0.5071, 0.4867, 0.4408],
        'std': [0.2675, 0.2565, 0.2761],
        'auto_augment': transforms.AutoAugmentPolicy.CIFAR10,
    },
    'cifar10': {
        'class_fn': torchvision.datasets.CIFAR10,
        'num_classes': 10,
        'native_size': 32,
        'img_size': 32,       # ← 使用原生 32x32，不上采样
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2470, 0.2435, 0.2616],
        'auto_augment': transforms.AutoAugmentPolicy.CIFAR10,
    },
    'svhn': {
        'class_fn': torchvision.datasets.SVHN,
        'num_classes': 10,
        'native_size': 32,
        'img_size': 32,       # ← 使用原生 32x32
        'mean': [0.4377, 0.4438, 0.4728],
        'std': [0.1980, 0.2010, 0.1970],
        'auto_augment': None,
    },
}


def get_available_datasets():
    """返回所有已注册的数据集名称列表 (list[str])"""
    return list(DATASET_REGISTRY.keys())


def _build_transforms(info: dict, is_train: bool):
    """
    根据数据集注册表中的信息，自动构建 Transform Pipeline。

    分辨率适配策略 (遵循学术标准):
      - native_size == img_size:  直接 RandomCrop(img_size, padding=4)，零插值失真
        这是 CIFAR 系列的标准做法，模型 Stem 层负责适配分辨率
      - native_size > img_size:   RandomResizedCrop(img_size)，标准长边裁剪做法

    Args:
        info (dict): 数据集注册信息
        is_train (bool): 是否为训练集
    Returns:
        transforms.Compose
    """
    native_size = info['native_size']
    img_size = info['img_size']
    mean = info['mean']
    std = info['std']

    transform_list = []

    if is_train:
        if native_size <= img_size:
            # 原生分辨率 ≤ 目标: 标准 CIFAR 增强 (RandomCrop + padding)
            transform_list.append(transforms.RandomCrop(img_size, padding=4))
        else:
            # 原生分辨率 > 目标: 使用 RandomResizedCrop 降采样 (标准学术做法)
            transform_list.append(transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)))

        transform_list.append(transforms.RandomHorizontalFlip())

        if info.get('auto_augment') is not None:
            transform_list.append(transforms.AutoAugment(info['auto_augment']))

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.25),
        ])
    else:
        # 验证集: 无数据增强，无分辨率变换
        if native_size > img_size:
            transform_list.append(transforms.Resize(int(img_size * 1.15)))
            transform_list.append(transforms.CenterCrop(img_size))

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    return transforms.Compose(transform_list)


def get_dataloaders(batch_size: int = 64,
                    num_workers: int = 4,
                    data_dir: str = './data',
                    dataset: str = 'cifar100'):
    """
    构建训练和验证数据加载器。

    Args:
        batch_size: 批大小
        num_workers: DataLoader 工作进程数
        data_dir: 数据集根目录
        dataset: 数据集名称 (需在 DATASET_REGISTRY 中注册)

    Returns:
        (train_loader, val_loader, train_dataset, dataset_info)
        dataset_info 包含 num_classes, img_size 等关键信息
    """
    dataset = dataset.lower()

    if dataset not in DATASET_REGISTRY:
        available = ', '.join(get_available_datasets())
        raise ValueError(
            f"未注册的数据集 '{dataset}'。当前支持: [{available}]\n"
            f"如需添加新数据集，请在 datasets.py 的 DATASET_REGISTRY 中新增条目。"
        )

    info = DATASET_REGISTRY[dataset]
    print(f"正在加载 {dataset.upper()} 数据集 "
          f"(原生 {info['native_size']}x{info['native_size']} → "
          f"模型输入 {info['img_size']}x{info['img_size']})...")

    train_transform = _build_transforms(info, is_train=True)
    val_transform = _build_transforms(info, is_train=False)

    ds_class = info['class_fn']

    if dataset == 'svhn':
        train_dataset = ds_class(root=data_dir, split='train', download=True, transform=train_transform)
        val_dataset = ds_class(root=data_dir, split='test', download=True, transform=val_transform)
    else:
        train_dataset = ds_class(root=data_dir, train=True, download=True, transform=train_transform)
        val_dataset = ds_class(root=data_dir, train=False, download=True, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    dataset_info = {
        'name': dataset,
        'num_classes': info['num_classes'],
        'img_size': info['img_size'],
    }

    return train_loader, val_loader, train_dataset, dataset_info
