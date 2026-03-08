import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from timm.utils import accuracy
import timm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import get_dataloaders
from custom_vit import CustomLightViT

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class Config:
    BATCH_SIZE = 128
    NUM_WORKERS = 4
    EPOCHS = 50
    LR = 5e-4
    WEIGHT_DECAY = 0.05
    NUM_CLASSES = 100
    LABEL_SMOOTHING = 0.1
    DATA_DIR = '/home/zjhao/bishe/my_tiny_vit/data'
    SAVE_DIR_BASE = './lightweight_saved'

os.makedirs(Config.SAVE_DIR_BASE, exist_ok=True)

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, scaler, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)

def evaluate(model, dataloader, device):
    model.eval()
    top1_acc = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
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
    print(f"[*] 准备在 {device} 上训练横向对比模型: {args.model}")

    train_loader, val_loader, _, _ = get_dataloaders(Config.BATCH_SIZE, Config.NUM_WORKERS, data_dir=Config.DATA_DIR)

    if args.model == 'custom_light_vit':
        model = CustomLightViT(num_classes=Config.NUM_CLASSES)
    else:
        # 统一关闭预训练以公平对比“从头开始训练”的轻量化能力
        model = timm.create_model(args.model, pretrained=False, num_classes=Config.NUM_CLASSES)
    
    model = model.to(device)

    save_dir = os.path.join(Config.SAVE_DIR_BASE, args.model)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'best_model.pth')
    log_file = os.path.join(save_dir, 'training_log.csv')

    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=Config.LR, epochs=Config.EPOCHS, steps_per_epoch=steps_per_epoch
    )
    scaler = GradScaler()
    best_acc = 0.0
    history = {'Epoch': [], 'Train_Loss': [], 'Val_Accuracy': []}

    print(f"[*] 开始以统一参数训练 {args.model} 共 {Config.EPOCHS} 个 Epochs...")
    
    for epoch in range(1, Config.EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device)
        val_acc = evaluate(model, val_loader, device)
        
        print(f"Epoch [{epoch}/{Config.EPOCHS}] | Train Loss: {train_loss:.4f} | Val Accuracy: {val_acc:.2f}%")
        
        history['Epoch'].append(epoch)
        history['Train_Loss'].append(train_loss)
        history['Val_Accuracy'].append(val_acc)
        pd.DataFrame(history).to_csv(log_file, index=False)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            
    print(f"[*] {args.model} 训练完毕，最高验证集准确率: {best_acc:.2f}%")

if __name__ == '__main__':
    main()
