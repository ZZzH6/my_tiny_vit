import os
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import torchvision.models as tv_models
from datetime import datetime

# Local imports
from config import Config
from datasets import get_dataloaders, get_available_datasets
from utils import setup_device, print_header, set_seed

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with autocast():
                outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

def parse_args():
    parser = argparse.ArgumentParser(description="Train a high-quality ResNet-50 teacher for Knowledge Distillation")
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=get_available_datasets(),
                        help=f"Choose dataset to train teacher on (default: cifar100)")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(Config.SEED)
    device = setup_device()
    
    dataset_name = args.dataset.upper()
    print_header(f"TRAINING HIGH-QUALITY RESNET-50 TEACHER FOR {dataset_name}")
    
    # 1. 准备数据 (通过 Dataset Registry 自动加载)
    train_loader, val_loader, _, dataset_info = get_dataloaders(
        Config.BATCH_SIZE, Config.NUM_WORKERS,
        data_dir=Config.DATA_DIR, dataset=args.dataset
    )
    num_classes = dataset_info['num_classes']
    print(f"[*] Dataset   : {dataset_name} ({num_classes} classes)")
    
    # 2. 构建模型
    print("Loading pre-trained ResNet50...")
    local_weight_path = './resnet50-0676ba61.pth'
    model = tv_models.resnet50(weights=None)
    if os.path.exists(local_weight_path):
        state_dict = torch.load(local_weight_path, map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        model = tv_models.resnet50(weights='DEFAULT')
        
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # === 优化: 替换 Stem 以适配 32x32 图像 (防止空间特征丢失过快) ===
    if dataset_info['img_size'] <= 64:
        # 原版是 7x7 stride=2, padding=3, 使得 32->16
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # 禁用随后的 maxpool 使得 16->8
        model.maxpool = nn.Identity()

    model = model.to(device)
    
    # 教师模型独立配置 (我们用标准的交叉熵，并不使用 Mixup)
    teacher_epochs = 40
    criterion = nn.CrossEntropyLoss()
    
    # === 优化: 权重衰减优化 (Bias 与 Norm 层解除抑制) ===
    # 区分 Head 层与 Base 层的参数, 以及是否应该应用 Weight Decay (一维特征通常是 Bias 或 Norm 层权/偏置)
    fc_params = []
    base_params = []
    fc_no_weight_decay = []
    base_no_weight_decay = []
    
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        is_fc = 'fc' in name
        no_decay = p.ndim <= 1 or name.endswith('.bias')
        
        if is_fc:
            fc_no_weight_decay.append(p) if no_decay else fc_params.append(p)
        else:
            base_no_weight_decay.append(p) if no_decay else base_params.append(p)
    
    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': 1e-4, 'weight_decay': 1e-4}, 
        {'params': base_no_weight_decay, 'lr': 1e-4, 'weight_decay': 0.0}, 
        {'params': fc_params, 'lr': 1e-3, 'weight_decay': 1e-4},
        {'params': fc_no_weight_decay, 'lr': 1e-3, 'weight_decay': 0.0}
    ])

    def lr_lambda(epoch):
        progress = epoch / float(max(1, teacher_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
        
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()
    
    best_acc = 0.0
    
    # Save into dedicated teacher directory (自动按数据集命名)
    os.makedirs('./teacher', exist_ok=True)
    save_path = f'./teacher/{dataset_name}_ResNet50_Teacher.pth'
    
    header = f"{'Epoch':^10} | {'Train Loss':^12} | {'Val Acc':^10} | {'Time(s)':^8} | {'LR':^10}"
    print(header)
    print("-" * len(header))
    
    start_time = datetime.now()
    
    for epoch in range(1, teacher_epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        
        current_lr = scheduler.get_last_lr()[1]  # print FC lr
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
            
        train_loss = running_loss / len(train_loader.dataset)
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()
        
        epoch_duration = time.time() - epoch_start
        epoch_str = f"[{epoch}/{teacher_epochs}]"
        
        best_mark = ""
        if val_acc > best_acc:
            best_acc = val_acc
            best_mark = " ★"
            torch.save(model.state_dict(), save_path)
            
        print(f"{epoch_str:^10} | {train_loss:^12.4f} | {val_acc:^10.2f} | {epoch_duration:^8.1f} | {current_lr:^10.2e}{best_mark}")

    end_time = datetime.now()
    total_duration = end_time - start_time
    minutes, seconds = divmod(total_duration.seconds, 60)
    
    print("\n======================================================================")
    print(f"[*] Finish Time      : {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[*] Total Time       : {minutes}m {seconds}s")
    print(f"[*] Best Teacher Acc : {best_acc:.2f}%")
    print(f"[*] Saved at         : {save_path}")
    print("======================================================================\n")

if __name__ == "__main__":
    main()
