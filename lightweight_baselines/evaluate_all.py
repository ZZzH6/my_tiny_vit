import os
import torch
import timm
from thop import profile, clever_format

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_vit import CustomLightViT

def evaluate_model_complexity(model, input_size=(1, 3, 224, 224), device='cpu'):
    model.eval()
    model.to(device)
    dummy_input = torch.randn(input_size).to(device)
    
    macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
    flops = macs * 2
    
    macs_str, params_str = clever_format([macs, params], "%.3f")
    flops_str, _ = clever_format([flops, params], "%.3f")
    
    return params_str, flops_str, params, flops

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*60)
    print(" 🌟 轻量化视觉模型复杂度横向对比 (Params & FLOPs) 🌟")
    print("="*60)
    
    models_to_test = [
        ("【你的核心贡献】CustomLightViT (6层, Dim=192, 插入CA)", lambda: CustomLightViT(num_classes=100)),
        ("【基线标杆1】MobileViT-XXS", lambda: timm.create_model('mobilevit_xxs', pretrained=False, num_classes=100)),
        ("【基线标杆2】DeiT-Tiny", lambda: timm.create_model('deit_tiny_patch16_224', pretrained=False, num_classes=100)),
        ("【极限天花板】ViT-Base (原版)", lambda: timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=100))
    ]
    
    results = []
    
    for name, model_fn in models_to_test:
        print(f"\n[*] 正在评估: {name} ...")
        try:
            model = model_fn()
            p_str, f_str, p, f = evaluate_model_complexity(model, device=device)
            results.append((name, p_str, f_str, p, f))
            print(f"    -> Params: {p_str} | FLOPs: {f_str}")
        except Exception as e:
            print(f"    ❌ 评估失败: {e}")
            
    # 输出 Markdown 表格供论文直接复制使用
    print("\n\n" + "-"*60)
    print("### 📚 建议写入论文的复杂度对比表格")
    print("| 模型架构方案 | 参数量 (Params) | 计算量 (FLOPs) |")
    print("| :--- | :--- | :--- |")
    for name, p_str, f_str, _, _ in results:
        # 为了表格好看，把名字截断或清理一下
        clean_name = name.split("】")[-1].strip()
        print(f"| {clean_name} | {p_str} | {f_str} |")
    print("-" * 60)

if __name__ == '__main__':
    main()
