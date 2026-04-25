# my_tiny_vit

本项目用于完成《基于 Transformer 的轻量化图像分类系统设计与实现》。

当前论文主线只保留两类实验：

- 方案A：`64 -> 224` 的标准 `timm` DeiT-Tiny baseline
- 方案B：`64 -> 112` 的轻量化主线，以及围绕该主线展开的结构消融和训练策略消融

当前不纳入本 README 的实验：

- 原生 `64 x 64` 输入实验
- 旧 teacher 驱动的早期 student / KD 实验
- 已确认无明显收益的中间搜索实验

## 当前主线

- baseline 必须使用 `timm`
- 224 主锚点模型：`deit_tiny_patch16_224`
- 112 主线模型：`deit_tiny_patch8_112`
- 主线评估协议统一使用 `use_imagenet_eval=false`
- 只有 `teacher_twostage_deitval` 保留为评估协议消融，不作为主结果
- student 蒸馏上游 teacher 固定使用 `strong_teacher_polish40`
- 当前最终 student 采用 `depth10 + DeiT hard distill + twostage refine`

## 数据目录

默认数据集目录为：

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

## 当前活动配置

| 配置 | 用途 |
|---|---|
| `configs/deit_tiny_baseline.yaml` | 224 输入标准 DeiT-Tiny baseline |
| `configs/deit_tiny_patch8_112_baseline.yaml` | 112 输入 baseline |
| `configs/deit_tiny_patch8_112_baseline_300ep.yaml` | 112 baseline 长训对照 |
| `configs/deit_tiny_patch8_112_overlap_patch12.yaml` | 112 主线结构改进：overlap patch embedding |
| `configs/deit_tiny_patch8_112_overlap_patch12_strong_teacher.yaml` | overlap teacher 强 recipe 版本 |
| `configs/deit_tiny_patch8_112_overlap_patch12_strong_teacher_polish40.yaml` | strong teacher 的 40 epoch 尾训版，当前最佳 teacher |
| `configs/deit_tiny_patch8_112_overlap_patch12_teacher_twostage.yaml` | 单配置两阶段复现版 |
| `configs/deit_tiny_patch8_112_student_depth10_logit_softkd.yaml` | depth10 student 的 soft logit KD 对照 |
| `configs/deit_tiny_patch8_112_student_depth10_deit_harddistill_same_recipe.yaml` | depth10 student 的 DeiT hard distill 公平对照 |
| `configs/deit_tiny_patch8_112_student_depth10_deit_harddistill_polish40.yaml` | depth10 student 的低正则尾训验证版 |
| `configs/deit_tiny_patch8_112_student_depth10_deit_harddistill_twostage.yaml` | 单配置两阶段复现版，当前最终 student |
| `configs/deit_tiny_patch8_112_student_depth9_logit_softkd.yaml` | depth9 soft KD 压缩消融 |
| `configs/deit_tiny_patch8_112_student_depth9_deit_harddistill_same_recipe.yaml` | depth9 hard KD 压缩消融 |

说明：

- 结构消融用到的 `LocalFFN / PreCNN / PrePatch` 配置已归档到 `configs/archive/112_ablation_retired/`，但结果仍可用于论文。
- `configs/deit_tiny_patch8_112_overlap_patch12_distilled_teacher.yaml` 已有结果，但使用的 teacher 不是当前最终 teacher，暂不纳入正文主结论。
- `student_depth10_deit_harddistill_polish40` 与 `student_depth10_deit_harddistill_twostage` 当前都达到 `79.41%`，正文建议优先使用 `twostage` 作为最终 student，因为复现路径更干净。

## 可写入论文的实验

### 1. 主结果

以下结果适合放入正文主表，统一使用 Tiny-ImageNet，统一按当前主线协议统计。

| 实验 | 输入 | Top-1 (%) | Params (M) | FLOPs (G) | 说明 |
|---|---:|---:|---:|---:|---|
| `deit_tiny_baseline` | 224 | 77.37 | 5.56 | 2.149 | 标准 `timm` DeiT baseline，方案A主锚点 |
| `patch8_112 baseline` | 112 | 79.46 | 5.45 | 2.106 | 方案B主 baseline |
| `patch8_112 + overlap patch12` | 112 | 79.73 | 5.50 | 2.124 | 当前最有效结构改进 |
| `overlap patch12 + strong recipe` | 112 | 79.77 | 5.50 | 2.124 | teacher 强化训练版本 |
| `strong teacher + polish40` | 112 | 80.11 | 5.50 | 2.124 | 当前最佳 teacher 结果 |
| `teacher_twostage` | 112 | 80.07 | 5.50 | 2.124 | 单配置两阶段复现版 |
| `student depth10 + hard KD + twostage` | 112 | 79.41 | 4.60 | 1.766 | 当前最终 student，较 112 baseline 明显降复杂度 |

主表建议写法：

- 224 标准 baseline 用于提供 DeiT 主锚点。
- 112 baseline 证明 `64 -> 112` 适配本身成立，且复杂度略低于 224 baseline。
- `overlap patch12` 是当前最有效的结构改进。
- `strong_teacher_polish40` 是当前最佳 teacher。
- `teacher_twostage` 说明两阶段训练可以合并为单配置复现流程。
- `student depth10 + hard KD + twostage` 是当前最终 student，在基本保持 112 baseline 精度的同时显著降低复杂度。

### 2. 112 主线结构消融

这组实验适合放入“结构消融”小节。为保证公平，建议只使用同一训练 recipe、同为 150 epoch 的结果。

| 结构 | 配置位置 | Top-1 (%) | Params (M) | FLOPs (G) | 相对 `patch8_112 baseline` |
|---|---|---:|---:|---:|---:|
| baseline | `configs/deit_tiny_patch8_112_baseline.yaml` | 79.46 | 5.45 | 2.106 | 0.00 |
| `+ LocalFFN` | `configs/archive/112_ablation_retired/deit_tiny_patch8_112_localffn.yaml` | 79.25 | 5.48 | 2.117 | -0.21 |
| `+ PreCNN` | `configs/archive/112_ablation_retired/deit_tiny_patch8_112_precnn.yaml` | 79.28 | 5.49 | 2.121 | -0.18 |
| `+ PreCNN + LocalFFN` | `configs/archive/112_ablation_retired/deit_tiny_patch8_112_precnn_localffn.yaml` | 79.34 | 5.52 | 2.132 | -0.12 |
| `+ PrePatch` | `configs/archive/112_ablation_retired/deit_tiny_patch8_112_prepatch.yaml` | 79.39 | 5.45 | 2.130 | -0.07 |
| `+ Overlap Patch12` | `configs/deit_tiny_patch8_112_overlap_patch12.yaml` | 79.73 | 5.50 | 2.124 | +0.27 |

这组结构消融的当前结论：

- 局部模块类改动并未稳定超过 baseline。
- `overlap patch embedding` 是唯一稳定带来增益的结构改动。
- 因此后续 teacher 主线固定采用 `overlap patch12`，不再继续堆叠无明显收益的局部模块。

### 3. 训练策略与评估协议消融

这组实验适合放入“训练策略消融”和“评估协议消融”。

| 实验 | Top-1 (%) | 结论 |
|---|---:|---|
| `patch8_112 baseline, 150ep` | 79.46 | 当前最合理的 112 baseline |
| `patch8_112 baseline, 300ep` | 78.61 | 继续长训无收益，甚至退化 |
| `overlap patch12` | 79.73 | 单做结构改进即可稳定增益 |
| `overlap patch12 + strong_teacher` | 79.77 | 强 recipe 仅小幅增益 |
| `strong_teacher_polish20` | 80.00 | 低正则尾训有效 |
| `strong_teacher_polish40` | 80.11 | 当前最佳结果 |
| `teacher_twostage` | 80.07 | 与独立尾训结果基本一致，适合作为单配置复现方案 |
| `teacher_twostage_deitval` | 79.53 | `use_imagenet_eval=true` 会明显拉低 Tiny-ImageNet 结果 |

当前可用于论文的结论：

- 112 baseline 不需要继续盲目长训到 300 epoch。
- teacher 的最优路线不是继续换结构，而是 `overlap patch12 + strong recipe + low-reg polish`。
- Tiny-ImageNet 主线评估应统一使用 `use_imagenet_eval=false`。

### 4. student 蒸馏与模型选择

这组实验适合放入“student 选择”和“蒸馏策略消融”小节。为保证公平，以下对照统一固定：

- 输入尺寸均为 `112`
- teacher 均为 `strong_teacher_polish40`
- depth9 / depth10 对照统一使用同一套 150 epoch student recipe
- 最终 student 再额外验证 `40 epoch` 低正则 refine 是否有效

| student 候选 | 蒸馏方式 | Top-1 (%) | Params (M) | FLOPs (G) | 说明 |
|---|---|---:|---:|---:|---|
| `depth9 + soft KD` | logit soft KD | 76.59 | 4.12 | 1.583 | 压缩更激进，但精度下降明显 |
| `depth9 + hard KD` | DeiT hard distill | 76.61 | 4.16 | 1.591 | 相比 soft KD 仅 +0.02，说明主要瓶颈是深度过低 |
| `depth10 + soft KD` | logit soft KD | 78.80 | 4.56 | 1.757 | 已稳定超过 224 baseline |
| `depth10 + hard KD` | DeiT hard distill | 78.94 | 4.60 | 1.766 | 同 recipe 下优于 soft KD |
| `depth10 + hard KD + polish40 / twostage` | DeiT hard distill + low-reg refine | 79.41 | 4.60 | 1.766 | 当前最终 student |

这组 student 蒸馏实验的当前结论：

- `depth10` 是当前更合理的 student 容量点；继续压到 `depth9` 会带来约 `2.8` 个点的精度损失。
- 在相同 teacher 与训练 recipe 下，`DeiT hard distill` 比 `soft logit KD` 更优，但单独增益不大，`depth10` 上仅提升 `0.14` 个点。
- 真正把 student 推到主线可用水平的是后续 `low-reg refine`，它让 `depth10 hard KD` 从 `78.94%` 提升到 `79.41%`。
- 当前最终 student `79.41 / 4.60M / 1.766G`，相比 `112 baseline` 仅低 `0.05` 个点，但参数量下降约 `15.6%`，FLOPs 下降约 `16.1%`。
- 相比 `224 baseline (77.37%)`，当前最终 student 仍高出 `2.04` 个点，说明学生蒸馏主线具备论文价值。

## 当前建议写入论文的结论

- `64 -> 112` 的 DeiT-Tiny 主线在略低复杂度下，已经稳定优于 `64 -> 224` 的标准 baseline。
- 在 112 主线的结构改造中，`overlap patch embedding` 是最有效且最干净的轻量化改进方向。
- 局部卷积类模块如 `LocalFFN`、`PreCNN`、`PrePatch` 没有在当前公平对照下提供稳定收益。
- 训练策略方面，`strong recipe + low-reg polish` 能把 112 teacher 提升到 `80.11%`。
- 两阶段单配置复现版 `teacher_twostage` 已能基本复现独立尾训效果，当前记录结果为 `80.07%`。
- student 主线方面，`depth10 + DeiT hard distill + low-reg refine` 可在 `4.60M / 1.766G` 下达到 `79.41%`，相比 `112 baseline` 几乎不掉点。
- `depth9` 虽然更轻，但在当前 teacher 与 recipe 下精度仅约 `76.6%`，不适合作为论文主推 student。

## 当前推荐引用的结果

如果论文只保留最核心的几条，建议引用：

- 224 baseline：`77.37 / 5.56M / 2.149G`
- 112 baseline：`79.46 / 5.45M / 2.106G`
- 112 overlap patch12：`79.73 / 5.50M / 2.124G`
- 最佳 teacher：`80.11 / 5.50M / 2.124G`
- 最终 student：`79.41 / 4.60M / 1.766G`

## 运行方式

### 1. 224 baseline

```bash
python -u scripts/train.py --config configs/deit_tiny_baseline.yaml
```

### 2. 112 baseline

```bash
python -u scripts/train.py --config configs/deit_tiny_patch8_112_baseline.yaml
```

### 3. overlap patch12 主线

```bash
python -u scripts/train.py --config configs/deit_tiny_patch8_112_overlap_patch12.yaml
```

### 4. 当前最佳 teacher

```bash
python -u scripts/train.py --config configs/deit_tiny_patch8_112_overlap_patch12_strong_teacher_polish40.yaml
```

### 5. 单配置两阶段复现版

```bash
python -u scripts/train.py --config configs/deit_tiny_patch8_112_overlap_patch12_teacher_twostage.yaml
```

### 6. depth10 student 蒸馏公平对照

```bash
python -u scripts/train.py --config configs/deit_tiny_patch8_112_student_depth10_logit_softkd.yaml
python -u scripts/train.py --config configs/deit_tiny_patch8_112_student_depth10_deit_harddistill_same_recipe.yaml
```

### 7. 最终 student 单配置复现版

```bash
python -u scripts/train.py --config configs/deit_tiny_patch8_112_student_depth10_deit_harddistill_twostage.yaml
```

如需补做更激进压缩消融，可运行：

```bash
python -u scripts/train.py --config configs/deit_tiny_patch8_112_student_depth9_logit_softkd.yaml
python -u scripts/train.py --config configs/deit_tiny_patch8_112_student_depth9_deit_harddistill_same_recipe.yaml
```

验证集评估：

```bash
python -u scripts/test.py \
  --config <config_path> \
  --checkpoint results/checkpoints/<date>/<run_id>_best.pt \
  --split val
```

测试集推理：

```bash
python -u scripts/test.py \
  --config <config_path> \
  --checkpoint results/checkpoints/<date>/<run_id>_best.pt \
  --split test
```

## 输出目录

所有训练与评估结果统一写入 `results/`：

- `results/checkpoints/`
- `results/logs/`
- `results/metrics/`
- `results/summary/`
- `results/eval/`
- `results/predictions/`

训练完成后统一输出：

- Top-1 / Top-5
- Params
- FLOPs
- best / last checkpoint
- log / summary / eval json
