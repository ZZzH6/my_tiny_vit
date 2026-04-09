# 实验发布规范

本文档定义本仓库中“正式实验结果”的发布口径，目标是让结果可追溯、可复现、可对比，避免训练产物混乱。

## 1. 结果层级

- `results/checkpoints/<date>/<run_id>_best.pt`：单次训练过程中的最佳权重。
- `results/checkpoints/<date>/<run_id>_last.pt`：单次训练过程中的最后状态，用于恢复训练。
- `results/models/<model_name>/best.pt`：某个模型当前全局最优权重，测试默认读取它。
- `results/models/<model_name>/best.json`：全局最优权重的来源记录。
- `results/eval/<date>/<config>_<run_id>_val.json`：有标签评估结果。
- `results/predictions/<date>/<config>_<run_id>_test.csv`：官方 test split 的预测导出。

## 2. 覆盖规则

- 单次 run 的 `*_best.pt` 和 `*_last.pt` 只属于该次 run，不互相覆盖。
- `results/models/<model_name>/best.pt` 只在当前 run 指标优于已有全局最优时才覆盖。
- 如果当前 run 没有超过已有最优，则保留旧的 model zoo 权重不变。

## 3. 论文口径

- 论文里的正式 baseline，优先使用 `results/models/<model_name>/best.pt` 对应结果。
- 论文里的 `Top-1 / Top-5` 若来自当前框架默认流程，应明确标注为 `val` 结果，不要写成独立 test result。
- 官方 Tiny-ImageNet `test` split 无标签时，只能做推理导出，不能直接汇报精度。
- 论文结果表中应报告单次最优值与多 seed 统计值，不要只报一次偶然跑出的最高值。
- 若使用恢复训练，必须在日志或 summary 中保留对应 `run_id` 和恢复来源。

## 4. 目录要求

- 训练日志统一放在 `results/logs/`
- 结构化指标统一放在 `results/metrics/`
- 汇总文档统一放在 `results/summary/`
- 评估结果统一放在 `results/eval/`
- test 推理结果统一放在 `results/predictions/`
- 模型级最优权重统一放在 `results/models/`

## 5. 默认使用方式

- 训练：

```bash
python -u scripts/train.py --config configs/deit_tiny_baseline.yaml
```

- 验证集评估：

```bash
python -u scripts/test.py --config configs/deit_tiny_baseline.yaml --split val
```

- 官方 test split 推理：

```bash
python -u scripts/test.py --config configs/deit_tiny_baseline.yaml --split test
```

- 恢复训练：

```bash
python -u scripts/train.py --config configs/deit_tiny_baseline.yaml --resume results/checkpoints/<run_id>_last.pt
```

## 6. 最低要求

- 每次正式训练必须产出 `best.pt`、`last.pt`、metrics、summary、val eval。
- 每个模型至少保留一个全局最优权重。
- 如需精确复现某次 run 的验证结果，应显式指定该 run 的 `*_best.pt`。
