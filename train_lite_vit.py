import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler

from models import CustomMobileViT
from config import Config
from datasets import get_dataloaders
from engine import train_one_epoch, evaluate
from utils import setup_device, compile_model_if_possible, save_model

def main():
    # 1. 硬件设备配置
    device = setup_device()

    # 2. 数据预处理与加载
    train_loader, val_loader, train_dataset, val_dataset = get_dataloaders(
        batch_size=Config.BATCH_SIZE, 
        num_workers=Config.NUM_WORKERS,
        data_dir=Config.DATA_DIR
    )

    # 3. 模型构建
    print("正在构建自定义的 CustomMobileViT 模型 (内含 Coordinate Attention)...")
    model = CustomMobileViT(num_classes=Config.NUM_CLASSES, pretrained=True, attention_type='coord')
    model = model.to(device)
    model = compile_model_if_possible(model)

    # 4. 训练组件设置
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING) 
    
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=Config.LR, 
        steps_per_epoch=len(train_loader), 
        epochs=Config.EPOCHS,
        pct_start=0.1 
    )

    # 去掉 'cuda' 参数，适配 PyTorch 1.x
    scaler = GradScaler()

    # 5. 核心训练与验证循环
    best_acc = 0.0 

    print("开始训练...")
    for epoch in range(Config.EPOCHS):
        # ---------- 训练阶段 ----------
        epoch_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device, len(train_dataset)
        )

        # ---------- 验证阶段 ----------
        epoch_acc = evaluate(model, val_loader, device)
        
        print(f"Epoch [{epoch+1}/{Config.EPOCHS}] | Train Loss: {epoch_loss:.4f} | Val Accuracy: {epoch_acc:.2f}%")

        # ---------- 保存机制 ----------
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            save_model(model, Config.SAVE_PATH, best_acc)

    print(f"训练完毕！最高验证集准确率为: {best_acc:.2f}%")

if __name__ == '__main__':
    main()