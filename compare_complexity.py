import os
import torch
import timm
from thop import profile, clever_format

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import CustomMobileViT

def evaluate_model_complexity(model, input_size=(1, 3, 224, 224), device='cpu'):
    """
    评估模型的参数量和计算量 (FLOPs)
    """
    model.eval()
    model.to(device)
    dummy_input = torch.randn(input_size).to(device)
    
    # 借助 thop 计算 FLOPs 和 Params
    macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
    
    # 将 MACs 转换为 FLOPs (通常 1 MAC = 2 FLOPs)
    flops = macs * 2
    
    # 格式化输出
    macs_str, params_str = clever_format([macs, params], "%.3f")
    flops_str, _ = clever_format([flops, params], "%.3f")
    
    return params_str, flops_str, params, flops

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"评估所用设备: {device}\n")
    print("="*50)
    print(" 模型复杂度横向对比报告 (Params & FLOPs)")
    print("="*50)
    
    # ===== 1. 评估轻量化自定义模型 =====
    try:
        print("\n[1] 正在评估自定义轻量化模型 (Custom MobileViT + Coordinate Attention)...")
        custom_model = CustomMobileViT(num_classes=100, pretrained=False, attention_type='coord')
        c_params_str, c_flops_str, c_p, c_f = evaluate_model_complexity(custom_model, device=device)
        print(f" -> 参数量 (Params): {c_params_str}")
        print(f" -> 计算量 (FLOPs):  {c_flops_str}")
    except Exception as e:
        print(f"评估自定义模型失败: {e}")
        c_p, c_f = None, None

    # ===== 2. 评估原版 ViT Baseline =====
    try:
        print("\n[2] 正在评估原版基线模型 (ViT-Base Patch16 224)...")
        baseline_model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=100)
        b_params_str, b_flops_str, b_p, b_f = evaluate_model_complexity(baseline_model, device=device)
        print(f" -> 参数量 (Params): {b_params_str}")
        print(f" -> 计算量 (FLOPs):  {b_flops_str}")
    except Exception as e:
        print(f"评估基线模型失败: {e}")
        b_p, b_f = None, None

    # ===== 3. 对比分析 =====
    if c_p and b_p:
        print("\n" + "="*50)
        print(" 【轻量化效果总结】")
        print(f" 相比原版 ViT-Base:")
        print(f" - 参数量缩减了: {b_p / c_p:.1f} 倍  ({(1 - c_p / b_p)*100:.2f}%)")
        print(f" - 计算量缩减了: {b_f / c_f:.1f} 倍  ({(1 - c_f / b_f)*100:.2f}%)")
        print("="*50)
        
    print("\n运行完毕。该数据可直接录入毕业论文的表格中进行展示。")

if __name__ == '__main__':
    main()
