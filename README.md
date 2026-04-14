# my_tiny_vit baseline

本仓库当前只保留最基础的 `timm` DeiT baseline。

## 当前保留内容

- 模型：`deit_tiny_patch16_224`
- 数据集：Tiny-ImageNet-200
- 输入方案：`64 -> 224`
- 数据增强：`RandomResizedCrop + RandomHorizontalFlip + Normalize`
- 训练入口：`scripts/train.py`
- 评估入口：`scripts/test.py`

不再保留：

- 强 baseline 配方
- 结构改进模型
- 与改进模型对应的配置文件和文档

## 数据目录

当前默认使用 `ImageFolder` 目录：

```text
dataset/tiny-imagenet-200/
  train/
    class_x/
      *.JPEG
  val/
    class_y/
      *.JPEG
  test/
    images/
      *.JPEG
```

## 训练

```bash
python -u scripts/train.py --config configs/deit_tiny_baseline.yaml
```

如需继续训练：

```bash
python -u scripts/train.py \
  --config configs/deit_tiny_baseline.yaml \
  --resume results/checkpoints/<date>/<run_id>_last.pt
```

## 评估

验证集：

```bash
python -u scripts/test.py \
  --config configs/deit_tiny_baseline.yaml \
  --checkpoint results/checkpoints/<date>/<run_id>_best.pt \
  --split val
```

测试集推理：

```bash
python -u scripts/test.py \
  --config configs/deit_tiny_baseline.yaml \
  --checkpoint results/checkpoints/<date>/<run_id>_best.pt \
  --split test
```

## 输出

训练和评估结果仍统一写入 `results/`：

- `results/checkpoints/`
- `results/logs/`
- `results/metrics/`
- `results/summary/`
- `results/eval/`
- `results/predictions/`

训练完成后会输出：

- `Top-1 / Top-5`
- `Params`
- `FLOPs`（若环境安装 `thop`）
- best / last checkpoint
