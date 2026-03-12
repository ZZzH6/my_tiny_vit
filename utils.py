import torch
import random
import numpy as np
import os

def setup_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

def set_seed(seed: int):
    """
    固定所有随机种子以确保实验可复现。
    (B, C, H, W) -> 统一随机性
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 牺牲部分性能以换取 100% 的可复现性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"[*] Random Seed: {seed} (Deterministic CUDNN: Enabled)")

def compile_model_if_possible(model):
    if hasattr(torch, 'compile') and os.name != 'nt':
        try:
            print("正在尝试使用 torch.compile 编译模型以加速...")
            model = torch.compile(model)
        except Exception as e:
            print(f"模型编译失败，回退到普通模式: {e}")
    else:
        print("当前环境不支持 torch.compile 或处于 Windows 下，使用普通模式。")
    return model

def save_model(model, save_path, acc):
    torch.save(model.state_dict(), save_path)
    print(f"🌟 新的最佳模型已保存 -> 准确率: {acc:.2f}% | 路径: {save_path}")

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
