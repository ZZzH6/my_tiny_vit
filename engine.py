import torch
from torch.cuda.amp import autocast

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, scaler, device, dataset_size):
    """训练模型的一个 Epoch。"""
    model.train()
    running_loss = 0.0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad(set_to_none=True) 
        
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        
        scaler.step(optimizer)
        scale_old = scaler.get_scale()
        scaler.update()
        
        if scale_old <= scaler.get_scale(): 
            scheduler.step()
        
        running_loss += loss.item() * inputs.size(0)
        
    epoch_loss = running_loss / dataset_size
    return epoch_loss

@torch.no_grad()
def evaluate(model, dataloader, device):
    """验证模型。"""
    model.eval()
    correct = 0
    total = 0
    
    # 使用 no_grad 代替 inference_mode，适配老版本环境
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
            
    epoch_acc = 100 * correct / total
    return epoch_acc
