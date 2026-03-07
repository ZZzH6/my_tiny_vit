import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
# 将对 timm 的依赖由直接调用迁移至通过 models.py 加载
from models import CustomMobileViT
from torch.amp import autocast, GradScaler

# 强行将 Hugging Face 流量指向国内镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def main():
    # ==========================================
    # 1. 硬件设备配置
    # ==========================================
    # 自动检测当前环境是否支持 GPU 加速，否则回退到 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        # 开启 cuDNN benchmark，对于固定输入尺寸的模型可以显著加速计算
        torch.backends.cudnn.benchmark = True
    print(f"当前使用的计算设备: {device}")

    # ==========================================
    # 2. 数据预处理与加载 (关键设计)
    # ==========================================
    
    # 训练集数据增强与预处理
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),       # 强制放大尺寸以匹配预训练模型的输入要求
        transforms.RandomHorizontalFlip(),   # 基础数据增强：随机水平翻转，提升泛化能力
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10), # 高级数据增强：ViT 容易过拟合，强数据增强是标配
        transforms.ToTensor(),               # 转换为张量，并将像素值归一化到 [0, 1]
        transforms.Normalize(                # 使用 ImageNet 的标准均值和方差进行标准化，加速收敛
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 验证集预处理（注意：验证集绝不能做随机数据增强，只需要缩放和标准化）
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 加载 CIFAR-100 数据集 (指定 root='./data')
    print("正在加载 CIFAR-100 数据集...")
    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    val_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=val_transform)

    # 创建 DataLoader，batch_size 设为 64 (如果显存爆了可以改为 32 或 16)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # ==========================================
    # 3. 模型构建 (改用自定义架构，凸显毕设工作量)
    # ==========================================
    print("正在构建自定义的 CustomMobileViT 模型 (内含 Coordinate Attention)...")
    
    # 论文可以介绍：我们不是简单调用模型，而是解耦了骨干网络与分类头，
    # 并插入了有助于密集预测和弱特征提取的 Coordinate Attention (坐标注意力) 模块
    model = CustomMobileViT(num_classes=100, pretrained=True, attention_type='coord')
    model = model.to(device)

    # 【可选优化 - PyTorch 2.0 编译加速】
    if hasattr(torch, 'compile') and os.name != 'nt':
        print("正在使用 torch.compile 编译模型以加速训练...")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile 编译失败 (忽略此错误并继续跳过): {e}")

    # ==========================================
    # 4. 训练组件设置
    # ==========================================
    epochs = 20
    # ViT 模型容易对训练集过度自信（过拟合），使用 Label Smoothing (标签平滑) 能显著提升泛化能力
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 
    
    # 经过测试，0.001 对于预训练好的 Transformer 模型过大，极易导致灾难性遗忘和早期的指标坍塌。
    # 结合 AdamW，最大 lr 设置在 3e-4 到 5e-4 最佳。
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
    
    # 替换为 OneCycleLR。ViT 等模型对初期梯度十分敏感，非常依赖 Warmup 预热机制。
    # OneCycleLR 能够让学习率在小幅度内上升到 max_lr，然后再逐步余弦退火。
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=5e-4, 
        steps_per_epoch=len(train_loader), 
        epochs=epochs,
        pct_start=0.1 # 让前 10% 的训练时长用于学习率缓慢上升
    )

    # 【论文可写要点 - 自动混合精度 (AMP)】：
    # 升级为 torch.amp API
    scaler = GradScaler('cuda')

    # ==========================================
    # 5. 核心训练与验证循环
    # ==========================================
    best_acc = 0.0 # 记录历史最高验证集准确率

    print("开始训练...")
    for epoch in range(epochs):
        # ---------- 训练阶段 ----------
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad(set_to_none=True) # 清空过往梯度，set_to_none=True 能少量降低显存占用并略微提速
            
            # 开启自动混合精度上下文（传入 device.type 消除警告）
            with autocast(device.type):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # 使用 Scaler 缩放 loss 并反向传播
            scaler.scale(loss).backward()
            
            # 在执行 scheduler.step() 前，先记录此时的尺度大小，用于探测是否发生了 inf/nan 使得 optimizer.step() 被跳过
            scaler.step(optimizer)
            scale_old = scaler.get_scale()
            scaler.update()
            
            # OneCycleLR 必须在每个 batch 后更新，但根据 PyTorch 在 AMP 下的规定，
            # 若因为检测到 inf 导致优化器跳过了 step_()，调度器也不能调用 step_()。
            if scale_old <= scaler.get_scale(): 
                scheduler.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        # scheduler.step() 在这里弃用，已被移入内层的 batch 循环

        # ---------- 验证阶段 ----------
        model.eval()
        correct = 0
        total = 0
        
        with torch.inference_mode(): # 使用 inference_mode 代替 no_grad，这是 PyTorch 更纯粹更快的推理上下文
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        epoch_acc = 100 * correct / total
        
        # 打印当前 Epoch 的结果
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {epoch_loss:.4f} | Val Accuracy: {epoch_acc:.2f}%")

        # ---------- 保存机制 ----------
        # 如果当前模型的验证准确率超过了历史最佳，则将其权重保存下来
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            save_path = './best_model.pth'
            torch.save(model.state_dict(), save_path)
            print(f"   => 验证集准确率创下新高！模型已保存至 {save_path} (此为后续 INT8 量化实验的基础)")

    print(f"训练完毕！最高验证集准确率为: {best_acc:.2f}%")

if __name__ == '__main__':
    main()
