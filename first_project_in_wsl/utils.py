import os
import torch

def setup_device():
    """设置硬件计算设备。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"当前使用的计算设备: {device}")
    return device


def compile_model_if_possible(model):
    """如果环境支持，尝试使用 torch.compile 加速模型。"""
    if hasattr(torch, 'compile') and os.name != 'nt':
        print("尝试使用 torch.compile...")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile 编译失败 (当前 PyTorch 版本可能不支持，跳过): {e}")
    return model


def save_model(model, path, acc):
    """保存模型权重。"""
    torch.save(model.state_dict(), path)
    print(f"   => 验证集准确率创下新高[{acc:.2f}%]！模型已保存至 {path} (此为后续 INT8 量化实验的基础)")
