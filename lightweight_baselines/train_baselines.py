import os
import argparse
import pandas as pd
import time
import math
from datetime import datetime
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from timm.utils import accuracy, ModelEmaV2
import timm

# Mixup/CutMix 增强 (防过拟合绝对核心)
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import get_dataloaders
from custom_vit import CustomLightViT

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class Config:
    BATCH_SIZE = 512  
    NUM_WORKERS = 16  
    EPOCHS = 200         
    LR = 1e-3         
    WEIGHT_DECAY = 0.1   
    NUM_CLASSES = 100
    LABEL_SMOOTHING = 0.1
    GRAD_CLIP_NORM = 1.0  
    EMA_DECAY = 0.9998    
    DATA_DIR = '/home/zjhao/bishe/my_tiny_vit/data'
    SAVE_DIR_BASE = './lightweight_saved'

    # Mixup 配置参数
    MIXUP_ALPHA = 0.8
    CUTMIX_ALPHA = 1.0
    PROB = 0.5       # 1.0 -> 0.5 (允许模型看一半干净样本，加快收敛)
    SWITCH_PROB = 0.5


os.makedirs(Config.SAVE_DIR_BASE, exist_ok=True)

def pad_string(s, length, align="left"):
    if align == "left":
        return str(s).ljust(length)
    elif align == "right":
        return str(s).rjust(length)
    return str(s).center(length)

def print_header(title):
    print("\n" + "="*70)
    print(pad_string(title, 70, "center"))
    print("="*70)

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, scaler, device, mixup_fn, ema=None):
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        # 应用 Mixup / CutMix
        if mixup_fn is not None:
            inputs, targets = mixup_fn(inputs, targets)

        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        
        # 梯度裁剪：防止 Transformer 训练中偶发的梯度爆炸
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.GRAD_CLIP_NORM)
        
        scaler.step(optimizer)
        scaler.update()
        
        # 更新 EMA 影子权重
        if ema is not None:
            ema.update(model)
            
        running_loss += loss.item() * inputs.size(0)
    
    # Custom LR Scheduler is Step-per-Epoch, unlike OneCycleLR which was Step-per-Batch
    scheduler.step()
    
    return running_loss / len(dataloader.dataset)

def evaluate(model, dataloader, device):
    model.eval()
    top1_acc = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with autocast():
                outputs = model(inputs)
            acc1, _ = accuracy(outputs, targets, topk=(1, 5))
            top1_acc += acc1.item() * inputs.size(0)
    return top1_acc / len(dataloader.dataset)

def parse_args():
    parser = argparse.ArgumentParser(description="Pytorch Lightweight Comparative Training")
    parser.add_argument('--model', type=str, required=True, 
                        choices=['custom_light_vit', 'mobilevit_xxs', 'deit_tiny_patch16_224'],
                        help="Choose which lightweight model to train")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True 
        
    print_header("LIGHTWEIGHT VISION TRANSFORMER TRAINING")
    print(f"[*] Model     : {args.model}")
    print(f"[*] Device    : {device}")
    print(f"[*] Batch     : {Config.BATCH_SIZE}")
    print(f"[*] Epochs    : {Config.EPOCHS}")
    print(f"[*] Peak LR   : {Config.LR}")
    print(f"[*] W-Decay   : {Config.WEIGHT_DECAY}")
    print(f"[*] Grad Clip : {Config.GRAD_CLIP_NORM}")
    print(f"[*] EMA Decay : {Config.EMA_DECAY}")
    print(f"[*] Mixup P   : {Config.PROB}")
    print("-" * 70)

    train_loader, val_loader, _, _ = get_dataloaders(Config.BATCH_SIZE, Config.NUM_WORKERS, data_dir=Config.DATA_DIR)

    if args.model == 'custom_light_vit':
        model = CustomLightViT(num_classes=Config.NUM_CLASSES)
    else:
        model = timm.create_model(args.model, pretrained=False, num_classes=Config.NUM_CLASSES)
    
    model = model.to(device)

    # 模型参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Params    : {total_params / 1e6:.2f} M")
    
    # 启用 timm 的完美 EMA (修复了 BatchNorm 运行均值无法更新的致命 bug)
    ema = ModelEmaV2(model, decay=Config.EMA_DECAY, device=device)
    print(f"[*] EMA       : Enabled (decay={Config.EMA_DECAY})")
    print("-" * 70)

    save_dir = os.path.join(Config.SAVE_DIR_BASE, args.model)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'best_model.pth')
    ema_save_path = os.path.join(save_dir, 'best_model_ema.pth')
    log_file = os.path.join(save_dir, 'training_log.csv')

    # Mixup 增强配置
    mixup_fn = Mixup(
        mixup_alpha=Config.MIXUP_ALPHA, 
        cutmix_alpha=Config.CUTMIX_ALPHA, 
        prob=Config.PROB, 
        switch_prob=Config.SWITCH_PROB, 
        mode='batch',
        label_smoothing=Config.LABEL_SMOOTHING, 
        num_classes=Config.NUM_CLASSES
    )
    
    # 使用 SoftTargetCrossEntropy 来匹配 Mixup 的软标签
    criterion = SoftTargetCrossEntropy()

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    
    # 自定义强力 LambdaLR 调度器 (Step per Epoch)
    def lr_lambda(epoch):
        warmup_epochs = 15
        hold_epochs = 150
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        elif epoch < hold_epochs:
            return 1.0
        else:
            progress = float(epoch - hold_epochs) / float(max(1, Config.EPOCHS - hold_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
            
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
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, mixup_fn, ema)
        
        # 评估原始模型
        val_acc = evaluate(model, val_loader, device)
        
        # 评估有效缓冲的 EMA 模型
        ema_val_acc = evaluate(ema.module, val_loader, device)
        
        epoch_duration = time.time() - epoch_start
        
        # 格式化输出
        epoch_str = f"[{epoch}/{Config.EPOCHS}]"
        best_mark = " ★" if ema_val_acc > best_ema_acc else ""
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
