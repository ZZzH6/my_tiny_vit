import os
import argparse
import pandas as pd
import time
import math
from datetime import datetime

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from timm.utils import ModelEmaV2
import timm
import torchvision.models as tv_models

from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy

# --------- Local Modules ---------
from config import Config
from datasets import get_dataloaders, get_available_datasets
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
                        choices=get_available_datasets(),
                        help=f"Choose dataset to train on (default: cifar100)")
    return parser.parse_args()

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

def get_kd_alpha(epoch: int):
    """
    分阶段蒸馏权重:
      - 前期保持基础 KD 权重
      - 中后期线性衰减
      - 最后 FINETUNE_EPOCHS 关闭 KD
    epoch 为 1-based
    """
    finetune_start = max(1, Config.EPOCHS - Config.FINETUNE_EPOCHS + 1)
    decay_start = max(1, int((finetune_start - 1) * Config.KD_DECAY_START_RATIO))

    if epoch >= finetune_start:
        return 0.0
    if epoch <= decay_start:
        return Config.KD_ALPHA

    decay_span = max(1, finetune_start - decay_start)
    progress = float(epoch - decay_start) / float(decay_span)
    return Config.KD_ALPHA * max(0.0, 1.0 - progress)

def is_finetune_epoch(epoch: int):
    finetune_start = max(1, Config.EPOCHS - Config.FINETUNE_EPOCHS + 1)
    return epoch >= finetune_start

def main():
    args = parse_args()
    set_seed(Config.SEED)
    device = setup_device()
        
    print_header("LIGHTWEIGHT VISION TRANSFORMER TRAINING")
    print(f"[*] Model     : {args.model}")
    print(f"[*] Dataset   : {args.dataset}")
    print(f"[*] Device    : {device}")
    print(f"[*] Batch     : {Config.BATCH_SIZE}")
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
    print("-" * 70)

    # 通过 Dataset Registry 动态获取数据集信息
    train_loader, val_loader, _, dataset_info = get_dataloaders(
        Config.BATCH_SIZE, Config.NUM_WORKERS,
        data_dir=Config.DATA_DIR, dataset=args.dataset
    )
    num_classes = dataset_info['num_classes']
    print(f"[*] Classes   : {num_classes}")

    if args.model == 'custom_light_vit':
        model = CustomLightViT(num_classes=num_classes)
    elif args.model == 'mobilevit_xxs':
        model = CustomMobileViT(num_classes=num_classes)
    else:
        model = timm.create_model(args.model, pretrained=False, num_classes=num_classes)
    
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Params    : {total_params / 1e6:.2f} M")
    
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

    mixup_fn = Mixup(
        mixup_alpha=Config.MIXUP_ALPHA, 
        cutmix_alpha=Config.CUTMIX_ALPHA, 
        prob=Config.PROB, 
        switch_prob=Config.SWITCH_PROB, 
        mode='batch',
        label_smoothing=Config.LABEL_SMOOTHING, 
        num_classes=num_classes
    )
    
    train_criterion = SoftTargetCrossEntropy()
    finetune_criterion = nn.CrossEntropyLoss(label_smoothing=Config.FINETUNE_LABEL_SMOOTHING)

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
    history = {'Epoch': [], 'Train_Loss': [], 'Val_Accuracy': [], 'EMA_Val_Accuracy': [], 'Time(s)': []}

    start_time = datetime.now()
    print_header(f"TRAINING INITIATED AT {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    header = f"{'Epoch':^10} | {'Train Loss':^12} | {'Val Acc':^10} | {'EMA Acc':^10} | {'Time(s)':^8} | {'LR':^10}"
    print(header)
    print("-" * len(header))

    for epoch in range(1, Config.EPOCHS + 1):
        epoch_start = time.time()
        
        current_lr = scheduler.get_last_lr()[0]
        
        current_kd_alpha = get_kd_alpha(epoch)
        current_mixup_fn = None if is_finetune_epoch(epoch) else mixup_fn
        current_criterion = finetune_criterion if is_finetune_epoch(epoch) else train_criterion

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
