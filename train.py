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

from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy

# --------- Local Modules ---------
from config import Config
from datasets import get_dataloaders
from engine import train_one_epoch, evaluate
from utils import setup_device, print_header
from custom_vit import CustomLightViT
from models import CustomMobileViT
# ---------------------------------

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def parse_args():
    parser = argparse.ArgumentParser(description="Pytorch Lightweight Comparative Training")
    parser.add_argument('--model', type=str, required=True, 
                        choices=['custom_light_vit', 'mobilevit_xxs', 'deit_tiny_patch16_224'],
                        help="Choose which lightweight model to train")
    return parser.parse_args()

def main():
    args = parse_args()
    device = setup_device()
        
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
    elif args.model == 'mobilevit_xxs':
        model = CustomMobileViT(num_classes=Config.NUM_CLASSES)
    else:
        model = timm.create_model(args.model, pretrained=False, num_classes=Config.NUM_CLASSES)
    
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Params    : {total_params / 1e6:.2f} M")
    
    ema = ModelEmaV2(model, decay=Config.EMA_DECAY, device=device)
    print(f"[*] EMA       : Enabled (decay={Config.EMA_DECAY})")
    print("-" * 70)

    save_dir = os.path.join(Config.SAVE_DIR_BASE, args.model)
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
        num_classes=Config.NUM_CLASSES
    )
    
    criterion = SoftTargetCrossEntropy()

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    
    def lr_lambda(epoch):
        warmup_epochs = 15
        hold_epochs = 100
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
