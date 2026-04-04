# my_tiny_vit baseline 说明

本仓库当前的 DeiT-Tiny 配置用于 Tiny-ImageNet-200 的正式 baseline 验证。

目标是把工程能力补完整，保证：

- 命令统一
- 训练可复现
- 结果可记录
- checkpoint 可恢复
- 指标可对比

不包含新的模型结构，也不包含蒸馏。

## 数据集目录要求

当前 loader 使用 `ImageFolder` 约定，目录需要是：

```text
dataset/tiny-imagenet-200/
  train/
    class_x/
      *.JPEG
  val/
    class_y/
      *.JPEG
```

注意：

- `val_annotations.txt` 不会被自动转换
- `train` 和 `val` 都必须是 `ImageFolder` 结构

## 训练命令

```bash
python -u scripts/train.py --config configs/deit_tiny_baseline.yaml
```

可恢复训练时增加 `--resume`：

```bash
python -u scripts/train.py --config configs/deit_tiny_baseline.yaml --resume results/checkpoints/<run_id>_last.pt
```

说明：

- `--resume` 必须指向 `*_last.pt`
- `*_best.pt` 只保存最佳权重，不用于恢复优化器状态

## 测试命令

命令风格和训练保持一致，只需要把 `train` 改成 `test`，再加上 checkpoint：

```bash
python -u scripts/test.py --config configs/deit_tiny_baseline.yaml --checkpoint results/checkpoints/<run_id>_best.pt --split val
```

说明：

- 默认 `split=val`
- 也支持 `split=train`，用于做训练集评估或调试

## 输出文件位置

所有实验产物统一保存到 `results/` 下：

- `results/logs/`
- `results/checkpoints/`
- `results/metrics/`
- `results/summary/`
- `results/eval/`

单次训练会生成：

- 日志文件
- `*_best.pt`
- `*_last.pt`
- 结构化 metrics 文件
- summary 文件
- 训练结束后的 eval 结果文件

## best / last checkpoint 区别

- `*_best.pt`
  - 保存验证集 `top1` 最好时对应的模型权重
  - 用于最终测试和论文汇报

- `*_last.pt`
  - 保存最后一轮训练状态
  - 包含 `model_state`、`optimizer_state`、`scaler_state`、`current_epoch`、`best_acc`、`best_epoch`
  - 用于恢复训练

## 当前 baseline 定位

当前 baseline 的定位是：

- 模型：`deit_tiny`
- 数据：`Tiny-ImageNet-200`
- 目标：形成一个工程上完整、可复现、可对比的正式 baseline

如果后续要做研究对比，建议直接基于这份 baseline 继续派生，不要在正式 baseline 上混入新的结构创新。

## 结果文件说明

训练完成后，推荐先看：

- `results/summary/.../*.md`
- `results/metrics/.../*.csv`
- `results/eval/.../*.json`

其中 summary 会包含：

- 配置摘要
- `best_val_acc`
- `best_epoch`
- `Params`
- `FLOPs`
- best / last checkpoint 路径
- log 路径
- metrics 路径
- eval 命令示例

## 复现说明

当前训练入口已经支持：

- seed
- deterministic
- AMP
- warmup + cosine lr
- gradient clipping
- early stopping
- best / last checkpoint

同一配置、同一 seed 下，训练过程和数据顺序应尽量保持一致。
