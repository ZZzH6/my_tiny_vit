import os
import json
import argparse
import pandas as pd
import time
import math
from datetime import datetime
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from timm.utils import ModelEmaV2
import timm
import torchvision.models as tv_models

from timm.data.mixup import Mixup

# --------- Local Modules ---------
from config import Config, reset_runtime_config, apply_runtime_overrides, get_active_config
from datasets import (
    get_dataloaders,
    get_available_datasets,
    get_dataset_strategy_note,
    get_dataset_training_overrides,
    normalize_dataset_name,
)
from engine import train_one_epoch, evaluate
from utils import setup_device, print_header, set_seed
from custom_vit import CustomLightViT
from models import CustomMobileViT
# ---------------------------------

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def parse_args():
    parser = argparse.ArgumentParser(description="Pytorch Lightweight Comparative Training")
    parser.add_argument('--model', type=str, required=True, 
                        choices=['custom_light_vit', 'mobilevit_xxs', 'deit_tiny_patch16_224'],
                        help="Choose which lightweight model to train")
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help=f"Choose dataset to train on. Available: {', '.join(get_available_datasets())}")
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight-decay', type=float, default=None)
    parser.add_argument('--warmup-epochs', type=int, default=None)
    parser.add_argument('--hold-epochs', type=int, default=None)
    parser.add_argument('--ema-decay', type=float, default=None)
    parser.add_argument('--kd-alpha', type=float, default=None)
    parser.add_argument('--kd-decay-start-ratio', type=float, default=None)
    parser.add_argument('--mixup-alpha', type=float, default=None)
    parser.add_argument('--cutmix-alpha', type=float, default=None)
    parser.add_argument('--mixup-prob', type=float, default=None)
    parser.add_argument('--switch-prob', type=float, default=None)
    parser.add_argument('--label-smoothing', type=float, default=None)
    parser.add_argument('--finetune-epochs', type=int, default=None)
    parser.add_argument('--finetune-lr', type=float, default=None)
    parser.add_argument('--finetune-label-smoothing', type=float, default=None)
    parser.add_argument('--embed-dim', type=int, default=None)
    parser.add_argument('--depth', type=int, default=None)
    parser.add_argument('--drop-rate', type=float, default=None)
    parser.add_argument('--drop-path-rate', type=float, default=None)
    return parser.parse_args()


def apply_overrides(args):
    override_map = {
        'DATA_DIR': args.data_dir,
        'BATCH_SIZE': args.batch_size,
        'NUM_WORKERS': args.num_workers,
        'EPOCHS': args.epochs,
        'LR': args.lr,
        'WEIGHT_DECAY': args.weight_decay,
        'WARMUP_EPOCHS': args.warmup_epochs,
        'HOLD_EPOCHS': args.hold_epochs,
        'EMA_DECAY': args.ema_decay,
        'KD_ALPHA': args.kd_alpha,
        'KD_DECAY_START_RATIO': args.kd_decay_start_ratio,
        'MIXUP_ALPHA': args.mixup_alpha,
        'CUTMIX_ALPHA': args.cutmix_alpha,
        'PROB': args.mixup_prob,
        'SWITCH_PROB': args.switch_prob,
        'LABEL_SMOOTHING': args.label_smoothing,
        'FINETUNE_EPOCHS': args.finetune_epochs,
        'FINETUNE_LR': args.finetune_lr,
        'FINETUNE_LABEL_SMOOTHING': args.finetune_label_smoothing,
    }
    applied_overrides = {}
    for key, value in override_map.items():
        if value is not None:
            setattr(Config, key, value)
            applied_overrides[key] = value
    return applied_overrides


def validate_runtime_config():
    if Config.EPOCHS <= 0:
        raise ValueError("Config.EPOCHS must be positive.")
    if Config.BATCH_SIZE <= 0:
        raise ValueError("Config.BATCH_SIZE must be positive.")
    if Config.NUM_WORKERS < 0:
        raise ValueError("Config.NUM_WORKERS must be non-negative.")
    if not 0.0 <= Config.PROB <= 1.0:
        raise ValueError("Config.PROB must be in [0, 1].")
    if not 0.0 <= Config.SWITCH_PROB <= 1.0:
        raise ValueError("Config.SWITCH_PROB must be in [0, 1].")
    if not 0.0 <= Config.LABEL_SMOOTHING < 1.0:
        raise ValueError("Config.LABEL_SMOOTHING must be in [0, 1).")
    if not 0.0 <= Config.FINETUNE_LABEL_SMOOTHING < 1.0:
        raise ValueError("Config.FINETUNE_LABEL_SMOOTHING must be in [0, 1).")
    if not 0.0 <= Config.KD_DECAY_START_RATIO <= 1.0:
        raise ValueError("Config.KD_DECAY_START_RATIO must be in [0, 1].")
    if Config.MIXUP_ALPHA < 0.0 or Config.CUTMIX_ALPHA < 0.0:
        raise ValueError("Mixup/CutMix alpha values must be non-negative.")
    if Config.WEIGHT_DECAY < 0.0:
        raise ValueError("Config.WEIGHT_DECAY must be non-negative.")
    if Config.WARMUP_EPOCHS < 0 or Config.HOLD_EPOCHS < 0:
        raise ValueError("Warmup/Hold epochs must be non-negative.")
    if Config.FINETUNE_EPOCHS < 0 or Config.FINETUNE_EPOCHS > Config.EPOCHS:
        raise ValueError("Config.FINETUNE_EPOCHS must be in [0, EPOCHS].")
    if Config.LR <= 0.0 or Config.FINETUNE_LR <= 0.0:
        raise ValueError("Learning rates must be positive.")

def split_weight_decay(model, weight_decay=0.05):
    """
    分离无需 weight decay 的参数（如 LayerNorm、BatchNorm2d、bias）
    和需要 weight decay 的参数（如 Conv2d.weight、Linear.weight 等 2D+ 张量）
    """
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.0},
        {'params': decay, 'weight_decay': weight_decay}
    ]


def get_custom_model_kwargs(args):
    model_kwargs = {}
    for key in ('embed_dim', 'depth', 'drop_rate', 'drop_path_rate'):
        value = getattr(args, key)
        if value is not None:
            model_kwargs[key] = value
    return model_kwargs


def get_finetune_start_epoch():
    if Config.FINETUNE_EPOCHS <= 0:
        return Config.EPOCHS + 1
    return max(1, Config.EPOCHS - Config.FINETUNE_EPOCHS + 1)


def clamp01(value: float):
    return max(0.0, min(1.0, value))


def cosine_interpolate(start_value: float, end_value: float, progress: float):
    progress = clamp01(progress)
    return end_value + 0.5 * (start_value - end_value) * (1.0 + math.cos(math.pi * progress))


def get_epoch_progress(epoch: int, start_epoch: int, end_epoch: int):
    if end_epoch <= start_epoch:
        return 1.0 if epoch >= end_epoch else 0.0
    if epoch < start_epoch:
        return 0.0
    if epoch >= end_epoch:
        return 1.0
    return float(epoch - start_epoch) / float(max(1, end_epoch - start_epoch))


def get_kd_alpha(epoch: int):
    """
    从指定比例的 epoch 开始平滑衰减 KD 权重，直到最后一个 epoch 归零。
    这样不会在 finetune 阶段产生目标函数的硬切换。
    """
    decay_start = max(1, int(Config.EPOCHS * Config.KD_DECAY_START_RATIO))
    if epoch < decay_start:
        return Config.KD_ALPHA
    progress = get_epoch_progress(epoch, decay_start, Config.EPOCHS)
    return cosine_interpolate(Config.KD_ALPHA, 0.0, progress)


def get_mixup_prob(epoch: int):
    finetune_start = get_finetune_start_epoch()
    if finetune_start > Config.EPOCHS:
        return Config.PROB
    progress = get_epoch_progress(epoch, finetune_start, Config.EPOCHS)
    return cosine_interpolate(Config.PROB, 0.0, progress)


def get_label_smoothing(epoch: int):
    finetune_start = get_finetune_start_epoch()
    if finetune_start > Config.EPOCHS:
        return Config.LABEL_SMOOTHING
    progress = get_epoch_progress(epoch, finetune_start, Config.EPOCHS)
    return cosine_interpolate(Config.LABEL_SMOOTHING, Config.FINETUNE_LABEL_SMOOTHING, progress)


def build_mixup_fn(num_classes: int, mixup_prob: float, label_smoothing: float):
    if mixup_prob <= 1e-8:
        return None
    return Mixup(
        mixup_alpha=Config.MIXUP_ALPHA,
        cutmix_alpha=Config.CUTMIX_ALPHA,
        prob=mixup_prob,
        switch_prob=Config.SWITCH_PROB,
        mode='batch',
        label_smoothing=label_smoothing,
        num_classes=num_classes
    )


def classification_loss(outputs, targets, label_smoothing=0.0):
    if torch.is_floating_point(targets):
        log_probs = F.log_softmax(outputs, dim=1)
        return torch.sum(-targets * log_probs, dim=1).mean()
    return F.cross_entropy(outputs, targets, label_smoothing=label_smoothing)


def format_overrides(overrides: dict):
    if not overrides:
        return "default profile"
    return ', '.join(f"{key}={value}" for key, value in sorted(overrides.items()))


def save_run_config(save_dir, args, model_kwargs, num_classes, total_params, dataset_strategy, manual_overrides):
    config_path = os.path.join(save_dir, 'run_config.json')
    run_config = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': args.model,
        'dataset': args.dataset,
        'num_classes': num_classes,
        'total_params': total_params,
        'config': get_active_config(),
        'dataset_strategy': dataset_strategy,
        'manual_overrides': manual_overrides,
        'custom_model_kwargs': model_kwargs,
    }
    with open(config_path, 'w', encoding='utf-8') as file:
        json.dump(run_config, file, indent=2, sort_keys=True)

def main():
    args = parse_args()
    args.dataset = normalize_dataset_name(args.dataset)
    reset_runtime_config()
    dataset_strategy = get_dataset_training_overrides(args.dataset)
    if dataset_strategy:
        apply_runtime_overrides(dataset_strategy)
    manual_overrides = apply_overrides(args)
    validate_runtime_config()
    dataset_strategy_note = get_dataset_strategy_note(args.dataset)

    set_seed(Config.SEED)
    device = setup_device()
    custom_model_kwargs = get_custom_model_kwargs(args)
        
    print_header("LIGHTWEIGHT VISION TRANSFORMER TRAINING")
    print(f"[*] Model     : {args.model}")
    print(f"[*] Dataset   : {args.dataset}")
    print(f"[*] Data Dir  : {Config.DATA_DIR}")
    print(f"[*] Strategy  : {dataset_strategy_note}")
    print(f"[*] Auto HP   : {format_overrides(dataset_strategy)}")
    print(f"[*] CLI HP    : {format_overrides(manual_overrides)}")
    print(f"[*] Device    : {device}")
    print(f"[*] Batch     : {Config.BATCH_SIZE}")
    print(f"[*] Workers   : {Config.NUM_WORKERS}")
    print(f"[*] Epochs    : {Config.EPOCHS}")
    print(f"[*] Peak LR   : {Config.LR}")
    print(f"[*] W-Decay   : {Config.WEIGHT_DECAY}")
    print(f"[*] Grad Clip : {Config.GRAD_CLIP_NORM}")
    print(f"[*] EMA Decay : {Config.EMA_DECAY}")
    print(f"[*] Warmup    : {Config.WARMUP_EPOCHS}")
    print(f"[*] Hold      : {Config.HOLD_EPOCHS}")
    print(f"[*] KD Temp   : {Config.KD_TEMPERATURE}")
    print(f"[*] KD Alpha  : {Config.KD_ALPHA}")
    print(f"[*] KD Decay  : {Config.KD_DECAY_START_RATIO}")
    print(f"[*] Finetune  : {Config.FINETUNE_EPOCHS}")
    print(f"[*] FT LR     : {Config.FINETUNE_LR}")
    print(f"[*] Mixup P   : {Config.PROB}")
    print(f"[*] Schedule  : Progressive KD / Mixup / Label Smoothing decay")
    print("-" * 70)

    # 通过 Dataset Registry 动态获取数据集信息
    train_loader, val_loader, _, dataset_info = get_dataloaders(
        Config.BATCH_SIZE, Config.NUM_WORKERS,
        data_dir=Config.DATA_DIR, dataset=args.dataset
    )
    num_classes = dataset_info['num_classes']
    print(f"[*] Classes   : {num_classes}")

    if args.model == 'custom_light_vit':
        model = CustomLightViT(num_classes=num_classes, **custom_model_kwargs)
    elif args.model == 'mobilevit_xxs':
        model = CustomMobileViT(num_classes=num_classes)
    else:
        model = timm.create_model(args.model, pretrained=False, num_classes=num_classes)
    
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Params    : {total_params / 1e6:.2f} M")
    if custom_model_kwargs:
        print(f"[*] Model Args : {custom_model_kwargs}")
    
    # === 知识蒸馏 Teacher ===
    teacher_path = f'./teacher/{args.dataset.upper()}_ResNet50_Teacher.pth'
    print(f"正在加载 High-Quality KD 教师模型 ({teacher_path})...")
    try:
        teacher_model = tv_models.resnet50(weights=None)
        teacher_model.fc = nn.Linear(teacher_model.fc.in_features, num_classes)
        
        # === 适配 32x32 图像 Stem ===
        if dataset_info['img_size'] <= 64:
            teacher_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            teacher_model.maxpool = nn.Identity()
        
        if os.path.exists(teacher_path):
            state_dict = torch.load(teacher_path, map_location='cpu')
            teacher_model.load_state_dict(state_dict)
            teacher_model = teacher_model.to(device)
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False
            print(f"[*] Teacher   : Loaded High-Quality ResNet50 for {args.dataset.upper()}. Ready for KD.")
        else:
            print(f"[!] Warning: {teacher_path} not found. Run train_teacher.py --dataset {args.dataset} first. Training WITHOUT KD.")
            teacher_model = None
    except Exception as e:
        print(f"[*] Teacher   : Failed to load ({e}). Training WITHOUT KD.")
        teacher_model = None
        
    ema = ModelEmaV2(model, decay=Config.EMA_DECAY, device=device)
    print(f"[*] EMA       : Enabled (decay={Config.EMA_DECAY})")
    print("-" * 70)

    save_dir = os.path.join(Config.SAVE_DIR_BASE, f"{args.dataset}_{args.model}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'best_model.pth')
    ema_save_path = os.path.join(save_dir, 'best_model_ema.pth')
    log_file = os.path.join(save_dir, 'training_log.csv')
    save_run_config(
        save_dir,
        args,
        custom_model_kwargs,
        num_classes,
        total_params,
        dataset_strategy,
        manual_overrides,
    )

    # == 应用 Weight Decay 的参数组解耦 ==
    optim_parameters = split_weight_decay(model, weight_decay=Config.WEIGHT_DECAY)
    optimizer = torch.optim.AdamW(optim_parameters, lr=Config.LR)
    
    def lr_lambda(epoch):
        warmup_epochs = Config.WARMUP_EPOCHS
        hold_epochs = Config.HOLD_EPOCHS
        finetune_epochs = Config.FINETUNE_EPOCHS
        finetune_start = max(0, Config.EPOCHS - finetune_epochs)
        decay_start = warmup_epochs + hold_epochs
        decay_end = max(decay_start + 1, finetune_start)
        finetune_factor = Config.FINETUNE_LR / Config.LR

        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        if epoch < decay_start:
            return 1.0
        if epoch >= finetune_start:
            return finetune_factor

        total_decay_epochs = max(1, decay_end - decay_start)
        progress = float(epoch - decay_start) / float(total_decay_epochs)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return finetune_factor + (1.0 - finetune_factor) * cosine
            
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    scaler = GradScaler()
    best_acc = 0.0
    best_ema_acc = 0.0
    history = {
        'Epoch': [],
        'Train_Loss': [],
        'Val_Accuracy': [],
        'EMA_Val_Accuracy': [],
        'Time(s)': [],
        'LR': [],
        'KD_Alpha': [],
        'Mixup_Prob': [],
        'Label_Smoothing': [],
    }

    start_time = datetime.now()
    print_header(f"TRAINING INITIATED AT {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    header = f"{'Epoch':^10} | {'Train Loss':^12} | {'Val Acc':^10} | {'EMA Acc':^10} | {'Time(s)':^8} | {'LR':^10}"
    print(header)
    print("-" * len(header))

    for epoch in range(1, Config.EPOCHS + 1):
        epoch_start = time.time()
        
        current_lr = scheduler.get_last_lr()[0]
        current_kd_alpha = get_kd_alpha(epoch)
        current_mixup_prob = get_mixup_prob(epoch)
        current_label_smoothing = get_label_smoothing(epoch)
        current_mixup_fn = build_mixup_fn(num_classes, current_mixup_prob, current_label_smoothing)
        current_criterion = partial(classification_loss, label_smoothing=current_label_smoothing)

        train_loss = train_one_epoch(
            model,
            train_loader,
            current_criterion,
            optimizer,
            scheduler,
            scaler,
            device,
            current_mixup_fn,
            ema,
            teacher_model,
            current_kd_alpha,
        )
        
        val_acc = evaluate(model, val_loader, device)
        ema_val_acc = evaluate(ema.module, val_loader, device)
        
        epoch_duration = time.time() - epoch_start
        
        epoch_str = f"[{epoch}/{Config.EPOCHS}]"
        best_mark = " ★" if val_acc > best_acc else ""
        print(f"{epoch_str:^10} | {train_loss:^12.4f} | {val_acc:^10.2f} | {ema_val_acc:^10.2f} | {epoch_duration:^8.1f} | {current_lr:^10.2e}{best_mark}")
        
        history['Epoch'].append(epoch)
        history['Train_Loss'].append(train_loss)
        history['Val_Accuracy'].append(val_acc)
        history['EMA_Val_Accuracy'].append(ema_val_acc)
        history['Time(s)'].append(epoch_duration)
        history['LR'].append(current_lr)
        history['KD_Alpha'].append(current_kd_alpha)
        history['Mixup_Prob'].append(current_mixup_prob)
        history['Label_Smoothing'].append(current_label_smoothing)
        pd.DataFrame(history).to_csv(log_file, index=False)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
        
        if ema_val_acc > best_ema_acc:
            best_ema_acc = ema_val_acc
            torch.save(ema.module.state_dict(), ema_save_path)
            
    end_time = datetime.now()
    total_duration = end_time - start_time
    hours, remainder = divmod(total_duration.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    print_header("TRAINING COMPLETED")
    print(f"[*] Finish Time      : {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[*] Total Time       : {hours}h {minutes}m {seconds}s")
    print(f"[*] Best Val Acc     : {best_acc:.2f}%")
    print(f"[*] Best EMA Val Acc : {best_ema_acc:.2f}%")
    print(f"[*] Model Saved      : {save_path}")
    print(f"[*] EMA Model Saved  : {ema_save_path}")
    print("=" * 70 + "\n")

if __name__ == '__main__':
    main()
