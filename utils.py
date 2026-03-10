import torch

def setup_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    return device

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
