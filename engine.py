import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from timm.utils import accuracy
from config import Config

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, scaler, device, mixup_fn, ema=None, teacher_model=None, kd_alpha=0.0):
    model.train()
    if teacher_model is not None:
        teacher_model.eval()

    running_loss = 0.0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        # 应用 Mixup / CutMix
        if mixup_fn is not None:
            inputs, targets = mixup_fn(inputs, targets)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            outputs = model(inputs)
            
            # 知识蒸馏逻辑 (Teacher-Student KD)
            if teacher_model is not None and kd_alpha > 0.0:
                with torch.no_grad():
                    teacher_outputs = teacher_model(inputs)
                
                # 真实标签的 Soft Lable Loss (由于 mixup, targets 已经是 shape [B, 100])
                loss_ce = criterion(outputs, targets)
                
                # 教师软标签 KL 散度
                temperature = Config.KD_TEMPERATURE
                loss_kd = F.kl_div(
                    F.log_softmax(outputs / temperature, dim=1),
                    F.softmax(teacher_outputs / temperature, dim=1),
                    reduction='batchmean'
                ) * (temperature * temperature)
                
                loss = (1.0 - kd_alpha) * loss_ce + kd_alpha * loss_kd
            else:
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
