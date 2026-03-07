import copy
import math
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.cuda.amp import autocast, GradScaler
from torch.distributions import Beta
from torch.utils.data import DataLoader
from einops import rearrange
import matplotlib.pyplot as plt
import time

# ==========================================
# 1. 核心模型定义 (保持你的创新结构)
# ==========================================
torch.backends.cudnn.benchmark = True


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class StochasticDepth(nn.Module):
    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x * random_tensor / keep_prob


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)
        for ema_buffer, buffer in zip(self.ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


class RobustLinearAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_scale = (dim // heads) ** -0.5
        self.pos_fix = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, h, w):
        b, n, d = x.shape
        res_pos = x.transpose(1, 2).reshape(b, d, h, w)
        pos = self.pos_fix(res_pos).flatten(2).transpose(1, 2)
        x = x + pos
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)
        context = torch.matmul(k.transpose(-1, -2), v)
        out = torch.matmul(q, context)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class RobustBlock(nn.Module):
    def __init__(self, in_c, out_c, d_model, drop_prob=0.0):
        super().__init__()
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, 1, 1, groups=in_c),
            nn.Conv2d(in_c, d_model, 1),
            nn.BatchNorm2d(d_model),
            nn.SiLU()
        )
        self.transformer = RobustLinearAttention(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.post_conv = nn.Conv2d(d_model, out_c, 1)
        self.drop_path = StochasticDepth(drop_prob) if drop_prob > 0 else nn.Identity()

    def forward(self, x):
        res = x
        x = self.pre_conv(x)
        b, c, h, w = x.shape
        identity = rearrange(x, 'b c h w -> b (h w) c')
        x_attn = self.norm(identity)
        x_attn = self.transformer(x_attn, h, w)
        x = identity + x_attn
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.post_conv(x)
        if x.shape == res.shape:
            return res + self.drop_path(x)
        return x

class ChallengeViT(nn.Module):
    def __init__(self, num_classes=100, drop_path_rate=0.1):
        super().__init__()
        num_blocks = 3
        drop_probs = [drop_path_rate * idx / max(num_blocks - 1, 1) for idx in range(num_blocks)]
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), 
            nn.BatchNorm2d(64),
            nn.SiLU()
        )
        self.stage1 = RobustBlock(64, 64, 64, drop_prob=drop_probs[0])
        self.down1 = nn.Conv2d(64, 128, 3, 2, 1)
        self.stage2 = RobustBlock(128, 128, 128, drop_prob=drop_probs[1])
        self.down2 = nn.Conv2d(128, 256, 3, 2, 1) 
        self.stage3 = RobustBlock(256, 256, 256, drop_prob=drop_probs[2])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        return self.head(x)


def mixup_data(inputs, targets, alpha):
    if alpha <= 0 or inputs.size(0) == 1:
        return inputs, targets, targets, 1.0, False
    lam = Beta(alpha, alpha).sample().item()
    lam = max(min(lam, 0.9), 0.1)
    perm = torch.randperm(inputs.size(0), device=inputs.device)
    mixed_inputs = lam * inputs + (1 - lam) * inputs[perm]
    targets_a = targets
    targets_b = targets[perm]
    return mixed_inputs, targets_a, targets_b, lam, True


@torch.no_grad()
def evaluate_accuracy(model, dataloader, device):
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(inputs)
        predictions = outputs.argmax(dim=1)
        correct += predictions.eq(labels).sum().item()
        total += labels.size(0)
    if was_training:
        model.train()
    return 100.0 * correct / total


def adjust_learning_rate(optimizer, base_lr, min_lr, epoch, warmup_epochs, total_epochs):
    if warmup_epochs > 0 and epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / max(1, warmup_epochs)
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        lr = min_lr + (base_lr - min_lr) * cosine
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# ==========================================
# 2. 训练、验证与测试逻辑
# ==========================================
def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        "batch_size": 128,
        "epochs": 100,
        "lr": 5e-4,
        "min_lr": 1e-5,
        "warmup_epochs": 5,
        "weight_decay": 0.05,
        "mixup_alpha": 0.2,
        "label_smoothing": 0.1,
        "drop_path_rate": 0.1,
        "ema_decay": 0.995,
        "use_amp": True,
        "grad_clip": 1.0,
        "seed": 42,
        "model_path": "best_model.pth",
        "device": device,
    }
    config["use_amp"] = config["use_amp"] and device.type == "cuda"
    config["num_workers"] = min(8, os.cpu_count() or 4)
    config["pin_memory"] = device.type == "cuda"

    set_seed(config["seed"])

    normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.8),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.25),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    def build_loader(dataset, shuffle):
        loader_kwargs = {
            "batch_size": config["batch_size"],
            "shuffle": shuffle,
            "num_workers": config["num_workers"],
            "pin_memory": config["pin_memory"],
        }
        if config["num_workers"] > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2
        return DataLoader(dataset, **loader_kwargs)

    trainloader = build_loader(trainset, True)
    testloader = build_loader(testset, False)

    model = ChallengeViT(num_classes=100, drop_path_rate=config["drop_path_rate"]).to(device)
    ema = ModelEMA(model, config["ema_decay"]) if config["ema_decay"] else None

    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scaler = GradScaler(enabled=config["use_amp"])

    history = {'train_loss': [], 'val_acc': [], 'lr': []}
    best_acc = 0.0

    print(f"开始训练，设备: {device}")
    start_time = time.time()

    for epoch in range(config["epochs"]):
        current_lr = adjust_learning_rate(
            optimizer,
            base_lr=config["lr"],
            min_lr=config["min_lr"],
            epoch=epoch,
            warmup_epochs=config["warmup_epochs"],
            total_epochs=config["epochs"]
        )
        history['lr'].append(current_lr)

        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            mixed_inputs, targets_a, targets_b, lam, use_mixup = mixup_data(inputs, labels, config["mixup_alpha"])

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=config["use_amp"]):
                outputs = model(mixed_inputs)
                if use_mixup:
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                else:
                    loss = criterion(outputs, targets_a)

            scaler.scale(loss).backward()
            if config["grad_clip"] > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

            if ema is not None:
                ema.update(model)

        avg_train_loss = running_loss / max(1, len(trainloader))
        eval_model = ema.ema_model if ema is not None else model
        val_acc = evaluate_accuracy(eval_model, testloader, device)

        history['train_loss'].append(avg_train_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(eval_model.state_dict(), config["model_path"])
            print(f"Epoch {epoch+1}: 新的最佳准确率 {best_acc:.2f}%，已保存模型。")

        print(f"Epoch {epoch+1}/{config['epochs']} | LR: {current_lr:.6f} | Loss: {avg_train_loss:.4f} | Acc: {val_acc:.2f}%")

    total_time = time.time() - start_time
    print(f"训练完成！总耗时: {total_time/60:.2f} 分钟，最佳准确率: {best_acc:.2f}%")

    evaluate_model(model, testloader, config)
    plot_history(history)

def evaluate_model(model, testloader, config):
    print("\n--- 正在加载最佳权重进行最终测试 ---")
    state_dict = torch.load(config["model_path"], map_location=config["device"])
    model.load_state_dict(state_dict)
    final_acc = evaluate_accuracy(model, testloader, config["device"])
    print(f"最终测试集准确率: {final_acc:.2f}%")

def plot_history(history):
    has_lr = 'lr' in history and history['lr']
    cols = 3 if has_lr else 2
    plt.figure(figsize=(16, 4))
    plt.subplot(1, cols, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.title('Loss Curve')
    plt.legend()
    
    plt.subplot(1, cols, 2)
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    if has_lr:
        plt.subplot(1, cols, 3)
        plt.plot(history['lr'], label='Learning Rate')
        plt.title('LR Schedule')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    print("训练曲线图已保存为 training_results.png")
    plt.show()

if __name__ == '__main__':
    run_experiment()
