import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import time
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

# ==========================================
# 原版 ViT 基线模型训练配置
# ==========================================


class Config:
    # 相比轻量化模型，原版 ViT 可能需要更小的 Batch Size 以防显存溢出
    BATCH_SIZE = 128  # 如果显存允许可以调大为 64
    NUM_WORKERS = 4
    EPOCHS = 20
    LR = 2e-5  # Reduced LR for fine-tuning a huge model
    WEIGHT_DECAY = 0.1  # Increased weight decay for stronger L2 regularization
    NUM_CLASSES = 100
    LABEL_SMOOTHING = 0.1
    SAVE_DIR = './baseline_saved'
    SAVE_PATH = os.path.join(SAVE_DIR, 'vit_base_patch16_224.pth')
    
    # 默认数据路径
    DATA_DIR = '/home/zjhao/bishe/my_tiny_vit/data'
    
    # 记录文件
    LOG_FILE = os.path.join(SAVE_DIR, 'vit_training_log.csv')
    PLOT_FILE = os.path.join(SAVE_DIR, 'vit_training_curve.png')

os.makedirs(Config.SAVE_DIR, exist_ok=True)

# ==========================================
# 训练与评估逻辑
# ==========================================

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
            
            # 关闭 AMP, 用 FP32 测准一些
            outputs = model(inputs)
            
            acc1, _ = accuracy(outputs, targets, topk=(1, 5))
            top1_acc += acc1.item() * inputs.size(0)
            
    return top1_acc / len(dataloader.dataset)

def plot_training_curves(log_file, plot_file):
    """根据保存的日志绘制 Loss 和 Accuracy 曲线"""
    try:
        df = pd.read_csv(log_file)
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Train Loss', color=color)
        ax1.plot(df['Epoch'], df['Train_Loss'], color=color, marker='o', label='Train Loss')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  
        color = 'tab:blue'
        ax2.set_ylabel('Val Accuracy (%)', color=color)  
        ax2.plot(df['Epoch'], df['Val_Accuracy'], color=color, marker='s', label='Val Accuracy')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  
        plt.title('Baseline ViT Training Progress on CIFAR-100')
        plt.grid(True, alpha=0.3)
        plt.savefig(plot_file, dpi=300)
        plt.close()
        print(f"=> 训练曲线图已保存至: {plot_file}")
    except Exception as e:
        print(f"绘制曲线图失败: {e}")

# ==========================================
# 主运行逻辑
# ==========================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前使用的计算设备: {device}")
    
    # 获取数据集
    train_loader, val_loader, _, _ = get_dataloaders(Config.BATCH_SIZE, Config.NUM_WORKERS, data_dir=Config.DATA_DIR)
    
    # ----- 构建原版 ViT 模型并本地加载预训练权重 -----
    print("正在构建基线 ViT-Base 模型 (vit_base_patch16_224)...")
    
    # 1. 关掉自动下载 (pretrained=False), 添加 Dropout 防过拟合
    model = timm.create_model(
        'vit_base_patch16_224', 
        pretrained=False, 
        num_classes=Config.NUM_CLASSES,
        drop_rate=0.1,         # 增加分类头的 Dropout
        attn_drop_rate=0.1     # 增加注意力机制的 Dropout
    )
    
    # 2. 指定你刚刚上传的本地权重路径 (请确保路径正确！)
    local_weights_path = '/home/zjhao/bishe/my_tiny_vit/baseline_vit/pytorch_model.bin'
    
    if os.path.exists(local_weights_path):
        print(f"[*] 找到本地权重，正在加载: {local_weights_path}")
        # 读取权重文件
        state_dict = torch.load(local_weights_path, map_location='cpu')
        
        # 处理可能嵌套的字典结构 (HuggingFace 下载的文件有时会包一层)
        if 'model' in state_dict:
            state_dict = state_dict['model']
            
        # ⚠️ 关键步骤：剔除预训练权重的分类头
        # 预训练模型通常是 1000 类或 21843 类，而我们的 CIFAR-100 是 100 类。
        # 如果不删掉原权重的分类头，尺寸不匹配会导致加载报错。
        for key in ['head.weight', 'head.bias']:
            if key in state_dict:
                del state_dict[key]
                print(f"[-] 已移除原权重的分类头参数: {key}")
                
        # 加载权重到模型中 (strict=False 允许某些层不匹配，比如我们刚删掉的分类头)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print("[*] 预训练权重加载成功！(分类头将随机初始化)")
    else:
        print(f"❌ 警告：未找到本地权重文件 {local_weights_path}！")
        print("模型将从零开始训练 (准确率可能会很低)。请先下载并上传权重文件。")
        exit() # 如果没找到权重，直接退出，防止白跑一趟

    model = model.to(device)
    
    # 打印参数量信息以便对比
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] 基线 ViT-Base 模型总参数量: {total_params / 1e6:.2f} M")
    
    # 定义损失函数、优化器和学习率调度器
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=Config.LR, 
        epochs=Config.EPOCHS, 
        steps_per_epoch=steps_per_epoch
    )
    
    scaler = GradScaler()
    best_acc = 0.0
    
    # 用于记录日志以便画图
    history = {'Epoch': [], 'Train_Loss': [], 'Val_Accuracy': []}

    print("开始训练原版 ViT Baseline...")
    try:
        for epoch in range(1, Config.EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device)
            val_acc = evaluate(model, val_loader, device)
            
            print(f"Epoch [{epoch}/{Config.EPOCHS}] | Train Loss: {train_loss:.4f} | Val Accuracy: {val_acc:.2f}%")
            
            # 记录数据
            history['Epoch'].append(epoch)
            history['Train_Loss'].append(train_loss)
            history['Val_Accuracy'].append(val_acc)
            
            # 实时保存日志 CSV 文件
            pd.DataFrame(history).to_csv(Config.LOG_FILE, index=False)
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), Config.SAVE_PATH)
                print(f"   => 验证集准确率创下新高[{best_acc:.2f}%]！基线模型已保存至 {Config.SAVE_PATH}")
                
        print(f"训练完毕！最高验证集准确率为: {best_acc:.2f}%")
        
        # 训练结束后自动画图
        plot_training_curves(Config.LOG_FILE, Config.PLOT_FILE)
        
    except KeyboardInterrupt:
        print("\n训练被手动中断。")
        plot_training_curves(Config.LOG_FILE, Config.PLOT_FILE)

if __name__ == '__main__':
    main()
