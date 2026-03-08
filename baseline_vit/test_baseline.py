import os
import torch
import timm
from timm.utils import accuracy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

import sys
# 把上级目录加入 sys.path, 因为我们需要引入 datasets.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import get_dataloaders

# 使用与训练时一致的配置
BATCH_SIZE = 128
NUM_WORKERS = 4
NUM_CLASSES = 100
DATA_DIR = '/home/zjhao/bishe/my_tiny_vit/data'
SAVE_DIR = '/home/zjhao/bishe/my_tiny_vit/baseline_saved'
SAVE_PATH = os.path.join(SAVE_DIR, 'vit_base_patch16_224.pth')

def test_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 评估所用计算设备: {device}")

    # ===== 获取测试集的数据加载器 =====
    # 在 CIFAR-100 中，train=False 获取的就是那 1 万张独占的测试集图像
    print("\n[*] 正在加载 CIFAR-100 测试集 (10,000 张图像)...")
    _, test_loader, _, test_dataset = get_dataloaders(BATCH_SIZE, NUM_WORKERS, data_dir=DATA_DIR)
    
    class_names = test_dataset.classes

    # ===== 构建并加载已训练模型 =====
    print("\n[*] 正在构建基线 ViT-Base 模型...")
    # 注意: drop_rate 不需要传递，因为我们在评估模式下，Dropout 默认关停
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=NUM_CLASSES)
    
    if os.path.exists(SAVE_PATH):
        print(f"[*] 模型文件存在，正在加载权重: {SAVE_PATH}")
        # 从该路径加载你的训练过的模型权重
        model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    else:
        print(f"❌ 警告：未找到你指定的模型权重文件 {SAVE_PATH}！")
        return

    model = model.to(device)
    model.eval()  # 必须设为 eval 模型，关闭 Dropout 和 Batch Norm 的训练行为

    print("\n[*] 开始在测试集上进行一轮完整推理，请稍候...")
    
    top1_acc_total = 0.0
    top5_acc_total = 0.0
    
    all_preds_list = []
    all_targets_list = []
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            # 模型前向传播计算结果
            outputs = model(inputs)
            
            # 使用 timm 封装的方法简便计算 Top-1 和 Top-5 准确率
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            top1_acc_total += acc1.item() * inputs.size(0)
            top5_acc_total += acc5.item() * inputs.size(0)
            
            # 为了绘制混淆矩阵，获取每张图片 Top-1 预测类的下标
            _, preds = torch.max(outputs, 1)
            all_preds_list.extend(preds.cpu().numpy())
            all_targets_list.extend(targets.cpu().numpy())
            
            # 每隔几个 batch 打印一下进度
            if (i+1) % 10 == 0:
                print(f"    - 推理进度: {i+1} / {len(test_loader)} Batches")
            
    total_samples = len(test_loader.dataset)
    final_top1_percent = top1_acc_total / total_samples
    final_top5_percent = top5_acc_total / total_samples
    
    print("\n" + "="*50)
    print(" 🏁 最终测试集 (Test Data) 评估数据总结")
    print("="*50)
    print(f" - 测试图片总数: {total_samples}")
    print(f" - Top-1 测试准确率: {final_top1_percent:.2f}%")
    print(f" - Top-5 测试准确率: {final_top5_percent:.2f}%")
    print("="*50)
    
    # ===== 生成 Scikit-Learn 的分类专业报表 =====
    print("\n[*] 正在根据结果生成分类量化报告 (包含 Precision, Recall 等)...")
    # 这对写论文极其极其加分！
    report = classification_report(all_targets_list, all_preds_list, target_names=class_names, digits=4)
    
    report_file = os.path.join(SAVE_DIR, 'vit_classification_report.txt')
    with open(report_file, 'w') as f:
        f.write("Baseline ViT-Base on CIFAR-100 Classification Report\n")
        f.write("="*70 + "\n")
        f.write(report)
    print(f" => ✔️ 分类报告成功导出至: {report_file}")
    
    # ===== 绘制 Seaborn 混淆矩阵 =====
    print("\n[*] 正在生成 CIFAR-100 类别混淆矩阵图片版...")
    cm_matrix = confusion_matrix(all_targets_list, all_preds_list)
    plt.figure(figsize=(24, 20))
    # annot=False，因为 100 个类别画全了字挤在一坨非常难看，用色彩浓度来展示即可
    sns.heatmap(cm_matrix, annot=False, cmap='Blues', fmt='d')
    plt.xlabel('Predicted Labels By Baseline ViT', fontsize=18)
    plt.ylabel('True Labels', fontsize=18)
    plt.title('TEST SET Confusion Matrix - Baseline ViT-Base', fontsize=22)
    
    cm_file = os.path.join(SAVE_DIR, 'vit_confusion_matrix.png')
    plt.savefig(cm_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f" => ✔️ 测试集混淆矩阵热图成功导出至: {cm_file}")
    
    print("\n测试脚本执行完毕。这些产生的数据报表及混淆矩阵将极大增强你最终论文的说服力！")

if __name__ == '__main__':
    test_model()
