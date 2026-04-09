# my_tiny_vit baseline 说明

本仓库当前的 DeiT-Tiny 配置用于 Tiny-ImageNet-200 的正式 baseline 验证。

目标是把工程能力补完整，保证：

- 命令统一
- 训练可复现
- 结果可记录
- checkpoint 可恢复
- 指标可对比

不包含新的模型结构，也不包含蒸馏。

当前默认配方保持 DeiT-Tiny 不变，只对训练工程做标准化优化：

- 方案A：Tiny-ImageNet `64 -> 224`
- baseline 模型：`deit_tiny_patch16_224`
- 派生模型：`deit_tiny_convstem`（仅替换 patch embedding 为轻量 conv stem）
- 第二步改进：`deit_tiny_convstem_localmixer`（conv stem + 浅层 local mixer + 深层 Transformer）
- `RandomResizedCrop + RandAugment + RandomErasing`
- `mixup(alpha=0.2) + cutmix + label smoothing`
- `50` epochs，`5` epochs warmup
- 不启用 early stopping

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
  test/
    images/
      *.JPEG
```

注意：

- `val_annotations.txt` 不会被自动转换
- `train` 和 `val` 都必须是 `ImageFolder` 结构
- 官方 `test` split 按 `test/images/*.JPEG` 读取，不要求标签目录

## 训练命令

baseline：

```bash
python -u scripts/train.py --config configs/deit_tiny_baseline.yaml
```

conv stem 改进模型：

```bash
python -u scripts/train.py --config configs/deit_tiny_convstem.yaml
```

conv stem + local mixer 改进模型：

```bash
python -u scripts/train.py --config configs/deit_tiny_convstem_localmixer.yaml
```

可恢复训练时增加 `--resume`：

```bash
python -u scripts/train.py --config <config> --resume results/checkpoints/<run_id>_last.pt
```

说明：

- `--resume` 必须指向 `*_last.pt`
- `*_best.pt` 只保存最佳权重，不用于恢复优化器状态

## 验证 / 测试命令

### 1. 验证集评估

默认会读取 `results/models/<model_name>/best.pt`，对 `val` 计算 `Top-1 / Top-5`：

```bash
python -u scripts/test.py --config <config> --split val
```

如果你要严格复现某一次 run 的结果，应显式指定该 run 的 `*_best.pt`：

```bash
python -u scripts/test.py \
  --config <config> \
  --checkpoint results/checkpoints/<date>/<run_id>_best.pt \
  --split val
```

### 2. 官方 test split 推理

`test` split 没有标签，脚本不会伪造精度，而是导出预测文件：

```bash
python -u scripts/test.py --config <config> --split test
```

也可以对某次 run 的 checkpoint 单独导出 test 预测：

```bash
python -u scripts/test.py \
  --config <config> \
  --checkpoint results/checkpoints/<date>/<run_id>_best.pt \
  --split test
```

说明：

- 默认会自动读取 `results/models/<model_name>/best.pt`
- `split=train` 和 `split=val` 是有标签评估，输出 `Top-1 / Top-5`
- `split=test` 是无标签推理，只导出预测结果，不输出精度
- 如果要指定某个历史 run，也可以额外加 `--checkpoint`

## 输出文件位置

所有实验产物统一保存到 `results/` 下：

- `results/logs/`
- `results/checkpoints/`
- `results/metrics/`
- `results/summary/`
- `results/eval/`
- `results/predictions/`
- `results/models/`

单次训练会生成：

- 日志文件
- `*_best.pt`
- `*_last.pt`
- 结构化 metrics 文件
- summary 文件
- 训练结束后的 val eval 结果文件

## best / last checkpoint 区别

- `*_best.pt`
  - 保存验证集 `top1` 最好时对应的模型权重
  - 用于最终测试和论文汇报

- `*_last.pt`
  - 保存最后一轮训练状态
  - 包含 `model_state`、`optimizer_state`、`scaler_state`、`current_epoch`、`best_acc`、`best_epoch`
  - 用于恢复训练

- `results/models/<model_name>/best.pt`
  - 保存该模型当前全局最优的权重
  - 训练结束后由程序自动比较并更新
  - 验证集评估和 test 推理默认读取这个文件

## 当前 baseline 定位

当前 baseline 的定位是：

- 模型：`deit_tiny`
- 数据：`Tiny-ImageNet-200`
- 目标：形成一个工程上完整、可复现、可对比、可直接进入论文表格的正式 baseline
- 训练口径：固定 50 轮，不做额外结构改动，不为 baseline 继续无休止调参

当前也提供第一步结构改进版本：

- 模型：`deit_tiny_convstem`
- 改动：仅将原始 patch embedding 替换为轻量 conv stem，再接回 DeiT-Tiny 主干
- 用途：作为后续消融与论文对比的第一步派生模型

当前还提供第二步结构改进版本：

- 模型：`deit_tiny_convstem_localmixer`
- 改动：保留 conv stem，并将浅层若干个 Transformer block 的标准 MHSA 替换为局部 token mixer，深层仍保留标准 Transformer
- 用途：让轻量化改动进一步作用到主干计算量，方便后续做消融和论文对比

如果后续要做研究对比，建议继续基于 baseline、conv stem、local mixer 三条明确模型线逐步派生，不要在正式 baseline 上混入新的结构创新。

实验发布规范见 [docs/experiment_release.md](/home/zjhao/bishe/my_tiny_vit/docs/experiment_release.md)。

## 结果文件说明

训练完成后，推荐先看：

- `results/summary/.../*.md`
- `results/metrics/.../*.csv`
- `results/eval/.../*.json`

其中 summary 会包含：

- 配置摘要
- `best_val_acc`
- `best_epoch`
- 模型级最优权重路径
- `Params`
- `FLOPs`
- best / last checkpoint 路径
- log 路径
- metrics 路径
- eval 命令示例
- test 推理命令示例

## 复现说明

当前训练入口已经支持：

- seed
- deterministic
- AMP
- warmup + cosine lr
- gradient clipping
- early stopping
- best / last checkpoint
- 模型级 best checkpoint 自动同步

同一配置、同一 seed 下，训练过程和数据顺序应尽量保持一致。

需要注意的实验口径：

- 训练期间选 best 使用的是 `val`
- `scripts/test.py --split val` 是验证集复评，不是独立测试集成绩
- `scripts/test.py --split test` 用于官方 test split 推理导出
