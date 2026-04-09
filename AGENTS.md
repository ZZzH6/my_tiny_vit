# AGENTS.md

## 1. Mission

本项目目标：完成《基于 Transformer 的轻量化图像分类系统设计与实现》。

核心任务：
1. 建立规范 DeiT baseline（使用 timm）
2. 在此基础上进行轻量化改进
3. 完成公平对比实验（精度 vs 复杂度）
4. 输出可用于论文的结果

禁止偏离“轻量化 + 图像分类”主线。

---

## 2. Baseline Rules（强制）

- baseline 必须使用 `timm`
- 默认模型：`deit_tiny_patch16_224`
- 默认输入：224×224（Tiny-ImageNet resize）
- 不使用官方 DeiT 源码重写 baseline
- 不允许为 baseline 进行无休止调参

baseline 达到以下条件即可停止优化：
- 能正常收敛
- 精度处于合理区间
- 可稳定复现

---

## 3. Core Principle

所有改动必须满足至少一个目标：
- 降低参数量
- 降低 FLOPs
- 提高推理速度
- 适配低分辨率输入（如 64×64）

若不满足以上任一目标，不应进行修改。

---

## 4. Allowed Modifications

允许的轻量化方向：

- 减少 depth
- 减少 embed_dim
- 减少 attention heads
- 减少 MLP ratio
- 修改 patch embedding（轻量化）
- CNN + Transformer 混合结构
- 小分辨率适配（img_size=64 / patch8）
- 蒸馏 / 剪枝 / 量化（可选）

禁止引入与课题无关的复杂结构。

---

## 5. Experiment Rules（强制）

所有实验必须保证：

- 相同数据集（Tiny-ImageNet）
- 相同划分
- 相同评价指标
- 尽量一致训练策略

必须输出：

- Top-1 accuracy
- 参数量（Params）
- FLOPs 或 MACs
- 推理时间 / FPS（可选）

禁止：
- 只报精度不报复杂度
- baseline 与改进模型使用不同训练策略

---

## 6. Tiny-ImageNet Rules

必须明确当前实验属于哪种：

- 方案A：64 → 224（baseline主线）
- 方案B：原生 64×64（扩展实验）

禁止混用或不标注输入尺寸。

---

## 7. Code Structure

必须遵守：

- configs/：配置文件
- models/：模型
- datasets/：数据处理
- trainer/ 或 engine/：训练逻辑
- tools/：统计脚本
- results/：日志与结果

禁止：
- 单文件堆所有代码
- 重复实现训练流程

---

## 8. Config Rules

以下内容必须可配置（禁止硬编码）：

- 数据路径
- batch size / epoch / lr
- 模型名称
- image size
- patch size
- 是否 pretrained
- 输出路径

必须支持：
通过修改 config 切换 baseline / 改进模型。

---

## 9. Working Style

- 先读代码再修改
- 每次只做一个改动
- 先跑通，再优化
- 不确定就查代码，不允许猜测

---

## 10. Git Rules

禁止：
- 删除已有实验结果
- 覆盖用户配置
- 修改无关代码

所有修改必须：
- 局部
- 可回退
- 可运行

---

## 11. Done Definition（完成标准）

任务完成必须满足：

- 代码可运行
- 结果可复现
- 输出路径明确
- 配置完整
- 改动目的清晰
- 可用于论文描述

否则不算完成。

---

## 12. Output Requirement

每次修改必须说明：

1. 改了什么
2. 为什么改
3. 如何运行
4. 输出什么结果
5. 对轻量化的影响

禁止模糊描述（如“已优化”、“效果更好”）。

---

## 13. Anti-Patterns（禁止行为）

- 过度调 baseline
- 只提升精度不降低复杂度
- 混乱实验对比
- 随意新增脚本
- 引入无关依赖
- 未理解代码直接重写
- 为“看起来强”而破坏公平性

---

## 14. Final Rule

所有工作必须服务于：

在标准 DeiT baseline 上，实现轻量化改进，
并证明在更低计算成本下保持可接受精度。
