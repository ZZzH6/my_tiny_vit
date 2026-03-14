"""
datasets.py - 多数据集注册表、数据增强策略与数据加载器
======================================================
统一管理:
  1. 数据集元信息 (类别数、原生分辨率、归一化统计量)
  2. 数据集级训练策略 (Mixup/CutMix/Warmup 等)
  3. 数据集级数据增强策略 (翻转、擦除、AutoAugment 等)
  4. 标准 torchvision 数据集与 Tiny-ImageNet 自定义读取逻辑
"""

import os
from urllib.parse import urlparse

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader, has_file_allowed_extension
from torchvision.datasets.utils import download_and_extract_archive


TINY_IMAGE_EXTENSIONS = ('.jpeg', '.jpg', '.png')
TINY_IMAGENET_ARCHIVE_NAME = 'tiny-imagenet-200.zip'
TINY_IMAGENET_MD5 = '90528d7ca1a48142e341f4ef8d21d0de'
TINY_IMAGENET_DEFAULT_URLS = [
    'https://scidata.sjtu.edu.cn/records/swpcp-01e87/files/tiny-imagenet-200.zip?download=1',
    'https://zenodo.org/records/10720917/files/tiny-imagenet-200.zip?download=1',
    'http://cs231n.stanford.edu/tiny-imagenet-200.zip',
]

DATASET_ALIASES = {
    'cifar10': 'cifar10',
    'cifar-10': 'cifar10',
    'cifar100': 'cifar100',
    'cifar-100': 'cifar100',
    'svhn': 'svhn',
    'tinyimagenet': 'tiny-imagenet',
    'tiny-imagenet': 'tiny-imagenet',
    'tiny_imagenet': 'tiny-imagenet',
}


def normalize_dataset_name(dataset: str):
    if not dataset:
        raise ValueError("数据集名称不能为空。")

    normalized = dataset.strip().lower().replace(' ', '').replace('_', '-')
    if normalized in DATASET_ALIASES:
        return DATASET_ALIASES[normalized]

    compact = normalized.replace('-', '')
    if compact in DATASET_ALIASES:
        return DATASET_ALIASES[compact]

    available = ', '.join(get_available_datasets())
    raise ValueError(f"未注册的数据集 '{dataset}'。当前支持: [{available}]")


def _resolve_tiny_imagenet_root(root: str):
    direct_root = os.path.abspath(root)
    candidates = [
        direct_root,
        os.path.join(direct_root, 'tiny-imagenet-200'),
        os.path.join(direct_root, 'tiny-imagenet'),
    ]

    for candidate in candidates:
        train_dir = os.path.join(candidate, 'train')
        val_dir = os.path.join(candidate, 'val')
        if os.path.isdir(train_dir) and os.path.isdir(val_dir):
            return candidate

    raise FileNotFoundError(
        "未找到 Tiny-ImageNet 数据集目录。"
        f" 请确认数据位于 '{direct_root}/tiny-imagenet-200'，"
        f" 或直接通过 --data-dir 指向包含 train/ 和 val/ 的 Tiny-ImageNet 根目录。"
    )


def _tiny_imagenet_is_available(root: str):
    try:
        _resolve_tiny_imagenet_root(root)
        return True
    except FileNotFoundError:
        return False


def _get_tiny_imagenet_download_root(root: str):
    direct_root = os.path.abspath(root)
    base_name = os.path.basename(os.path.normpath(direct_root))
    if base_name in {'tiny-imagenet-200', 'tiny-imagenet'}:
        return os.path.dirname(direct_root)
    return direct_root


def _get_tiny_imagenet_download_urls():
    custom_urls = os.environ.get('TINY_IMAGENET_URLS')
    if not custom_urls:
        return list(TINY_IMAGENET_DEFAULT_URLS)

    urls = [item.strip() for item in custom_urls.split(',') if item.strip()]
    if not urls:
        return list(TINY_IMAGENET_DEFAULT_URLS)
    return urls


def _sanitize_download_filename(url: str):
    parsed = urlparse(url)
    file_name = os.path.basename(parsed.path)
    if file_name.endswith('.zip'):
        return file_name
    return TINY_IMAGENET_ARCHIVE_NAME


def ensure_dataset_available(dataset: str, data_dir: str):
    dataset_key = normalize_dataset_name(dataset)
    if dataset_key != 'tiny-imagenet':
        return

    if _tiny_imagenet_is_available(data_dir):
        return

    download_root = _get_tiny_imagenet_download_root(data_dir)
    os.makedirs(download_root, exist_ok=True)

    errors = []
    print("[*] Tiny-ImageNet 未检测到，本次将自动下载并解压。")
    print("[*] 下载源顺序: 国内镜像优先，失败后自动回退到其他镜像/官方源。")

    for url in _get_tiny_imagenet_download_urls():
        archive_name = _sanitize_download_filename(url)
        print(f"[*] 正在尝试下载 Tiny-ImageNet: {url}")
        try:
            download_and_extract_archive(
                url=url,
                download_root=download_root,
                extract_root=download_root,
                filename=archive_name,
                md5=TINY_IMAGENET_MD5,
                remove_finished=False,
            )
            if _tiny_imagenet_is_available(data_dir):
                print("[*] Tiny-ImageNet 下载并解压完成。")
                return
            errors.append(f"{url} -> 下载结束但未发现 tiny-imagenet-200 目录")
        except Exception as exc:
            errors.append(f"{url} -> {exc}")

    error_message = '\n'.join(f"  - {item}" for item in errors)
    raise RuntimeError(
        "Tiny-ImageNet 自动下载失败，所有镜像源均不可用。\n"
        "你可以稍后重试，或通过环境变量 TINY_IMAGENET_URLS 指定新的下载地址。\n"
        f"尝试记录:\n{error_message}"
    )


class TinyImageNetDataset(VisionDataset):
    """
    读取 Tiny-ImageNet-200。

    目录结构支持:
      - data_dir/tiny-imagenet-200/{train,val}
      - 直接把 data_dir 指向 tiny-imagenet-200 根目录
    """
    def __init__(self, root, split='train', transform=None):
        self.dataset_root = _resolve_tiny_imagenet_root(root)
        super().__init__(self.dataset_root, transform=transform)

        if split not in {'train', 'val'}:
            raise ValueError(f"Tiny-ImageNet 仅支持 'train' 或 'val'，收到: {split}")

        self.split = split
        self.loader = default_loader
        self.classes = self._load_wnids()
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        self.samples = self._build_samples()
        self.targets = [target for _, target in self.samples]
        self.imgs = self.samples

        if not self.samples:
            raise RuntimeError(f"Tiny-ImageNet {split} split 中未找到任何图像样本。")

    def _load_wnids(self):
        wnids_path = os.path.join(self.dataset_root, 'wnids.txt')
        if not os.path.isfile(wnids_path):
            raise FileNotFoundError(f"缺少 Tiny-ImageNet 类别索引文件: {wnids_path}")

        with open(wnids_path, 'r', encoding='utf-8') as file:
            return [line.strip() for line in file if line.strip()]

    def _build_samples(self):
        if self.split == 'train':
            return self._build_train_samples()
        return self._build_val_samples()

    def _build_train_samples(self):
        train_dir = os.path.join(self.dataset_root, 'train')
        samples = []

        for class_name in self.classes:
            images_dir = os.path.join(train_dir, class_name, 'images')
            if not os.path.isdir(images_dir):
                raise FileNotFoundError(f"Tiny-ImageNet 训练目录缺失: {images_dir}")

            for image_name in sorted(os.listdir(images_dir)):
                if not has_file_allowed_extension(image_name.lower(), TINY_IMAGE_EXTENSIONS):
                    continue
                samples.append((os.path.join(images_dir, image_name), self.class_to_idx[class_name]))

        return samples

    def _build_val_samples(self):
        val_dir = os.path.join(self.dataset_root, 'val')
        image_dir = os.path.join(val_dir, 'images')
        annotation_path = os.path.join(val_dir, 'val_annotations.txt')

        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Tiny-ImageNet 验证图片目录缺失: {image_dir}")
        if not os.path.isfile(annotation_path):
            raise FileNotFoundError(f"Tiny-ImageNet 验证标注文件缺失: {annotation_path}")

        samples = []
        with open(annotation_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                image_name, class_name = parts[0], parts[1]
                image_path = os.path.join(image_dir, image_name)
                if not os.path.isfile(image_path):
                    raise FileNotFoundError(f"Tiny-ImageNet 验证图片缺失: {image_path}")
                samples.append((image_path, self.class_to_idx[class_name]))

        return samples

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        image = self.loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.samples)


DATASET_REGISTRY = {
    'cifar100': {
        'class_fn': torchvision.datasets.CIFAR100,
        'num_classes': 100,
        'native_size': 32,
        'img_size': 32,
        'mean': [0.5071, 0.4867, 0.4408],
        'std': [0.2675, 0.2565, 0.2761],
        'auto_augment': transforms.AutoAugmentPolicy.CIFAR10,
        'use_horizontal_flip': True,
        'random_erasing_prob': 0.25,
        'train_config': {},
        'teacher_config': {},
        'strategy_note': '默认小图策略: 使用原生 32x32 输入，保守 Mixup，关闭 CutMix。',
    },
    'cifar10': {
        'class_fn': torchvision.datasets.CIFAR10,
        'num_classes': 10,
        'native_size': 32,
        'img_size': 32,
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2470, 0.2435, 0.2616],
        'auto_augment': transforms.AutoAugmentPolicy.CIFAR10,
        'use_horizontal_flip': True,
        'random_erasing_prob': 0.15,
        'train_config': {
            'LABEL_SMOOTHING': 0.02,
            'MIXUP_ALPHA': 0.10,
            'PROB': 0.10,
            'KD_ALPHA': 0.30,
            'FINETUNE_EPOCHS': 20,
        },
        'teacher_config': {},
        'strategy_note': '类别更少，降低标签混合与擦除强度，加快 clean fine-tune。',
    },
    'svhn': {
        'class_fn': torchvision.datasets.SVHN,
        'num_classes': 10,
        'native_size': 32,
        'img_size': 32,
        'mean': [0.4377, 0.4438, 0.4728],
        'std': [0.1980, 0.2010, 0.1970],
        'auto_augment': None,
        'use_horizontal_flip': False,
        'random_erasing_prob': 0.10,
        'train_config': {
            'LABEL_SMOOTHING': 0.0,
            'MIXUP_ALPHA': 0.0,
            'CUTMIX_ALPHA': 0.0,
            'PROB': 0.0,
            'SWITCH_PROB': 0.0,
            'KD_ALPHA': 0.20,
            'FINETUNE_EPOCHS': 10,
        },
        'teacher_config': {},
        'strategy_note': '数字识别任务，禁用水平翻转与 Mixup/CutMix，避免语义破坏。',
    },
    'tiny-imagenet': {
        'class_fn': TinyImageNetDataset,
        'num_classes': 200,
        'native_size': 64,
        'img_size': 64,
        'mean': [0.4802, 0.4481, 0.3975],
        'std': [0.2302, 0.2265, 0.2262],
        'auto_augment': transforms.AutoAugmentPolicy.IMAGENET,
        'use_horizontal_flip': True,
        'random_erasing_prob': 0.10,
        'train_config': {
            'BATCH_SIZE': 128,
            'WEIGHT_DECAY': 0.05,
            'LABEL_SMOOTHING': 0.10,
            'WARMUP_EPOCHS': 20,
            'MIXUP_ALPHA': 0.20,
            'CUTMIX_ALPHA': 1.0,
            'PROB': 0.40,
            'SWITCH_PROB': 0.50,
            'KD_ALPHA': 0.30,
            'KD_DECAY_START_RATIO': 0.70,
            'FINETUNE_EPOCHS': 30,
            'FINETUNE_LR': 1e-4,
        },
        'teacher_config': {
            'BATCH_SIZE': 128,
        },
        'strategy_note': '64x64 原生输入，减小 batch 并强化 Warmup、Mixup/CutMix 与正则化。',
    },
}


def get_available_datasets():
    """返回所有已注册的数据集名称列表 (canonical names)。"""
    return list(DATASET_REGISTRY.keys())


def get_dataset_info(dataset: str):
    dataset_key = normalize_dataset_name(dataset)
    return DATASET_REGISTRY[dataset_key]


def get_dataset_training_overrides(dataset: str):
    return dict(get_dataset_info(dataset).get('train_config', {}))


def get_dataset_teacher_overrides(dataset: str):
    return dict(get_dataset_info(dataset).get('teacher_config', {}))


def get_dataset_strategy_note(dataset: str):
    return get_dataset_info(dataset).get('strategy_note', '')


def _build_transforms(info: dict, is_train: bool):
    """
    根据数据集注册信息自动构建 Transform Pipeline。

    分辨率适配原则:
      - native_size == img_size: 使用 RandomCrop + padding，避免不必要上采样
      - native_size > img_size: 使用 RandomResizedCrop 做标准降采样
    """
    native_size = info['native_size']
    img_size = info['img_size']
    mean = info['mean']
    std = info['std']

    transform_list = []

    if is_train:
        if native_size <= img_size:
            transform_list.append(transforms.RandomCrop(img_size, padding=4))
        else:
            transform_list.append(transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)))

        if info.get('use_horizontal_flip', True):
            transform_list.append(transforms.RandomHorizontalFlip())

        if info.get('auto_augment') is not None:
            transform_list.append(transforms.AutoAugment(info['auto_augment']))

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        random_erasing_prob = info.get('random_erasing_prob', 0.25)
        if random_erasing_prob > 0.0:
            transform_list.append(transforms.RandomErasing(p=random_erasing_prob))
    else:
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

    Returns:
        (train_loader, val_loader, train_dataset, dataset_info)
    """
    dataset_key = normalize_dataset_name(dataset)
    info = DATASET_REGISTRY[dataset_key]
    ensure_dataset_available(dataset_key, data_dir)

    print(
        f"正在加载 {dataset_key.upper()} 数据集 "
        f"(原生 {info['native_size']}x{info['native_size']} -> "
        f"模型输入 {info['img_size']}x{info['img_size']})..."
    )

    train_transform = _build_transforms(info, is_train=True)
    val_transform = _build_transforms(info, is_train=False)
    ds_class = info['class_fn']

    if dataset_key == 'svhn':
        train_dataset = ds_class(root=data_dir, split='train', download=True, transform=train_transform)
        val_dataset = ds_class(root=data_dir, split='test', download=True, transform=val_transform)
    elif dataset_key == 'tiny-imagenet':
        train_dataset = ds_class(root=data_dir, split='train', transform=train_transform)
        val_dataset = ds_class(root=data_dir, split='val', transform=val_transform)
    else:
        train_dataset = ds_class(root=data_dir, train=True, download=True, transform=train_transform)
        val_dataset = ds_class(root=data_dir, train=False, download=True, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    dataset_info = {
        'name': dataset_key,
        'num_classes': info['num_classes'],
        'img_size': info['img_size'],
        'native_size': info['native_size'],
        'strategy_note': info.get('strategy_note', ''),
    }

    return train_loader, val_loader, train_dataset, dataset_info
