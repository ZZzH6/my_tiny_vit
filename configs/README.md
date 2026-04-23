# Configs Index

当前 `configs/` 根目录只保留论文主线和仍需复现实验的配置；已确认无收益或仅作为中间试验的配置统一移动到 `configs/archive/`。

## Root Configs

- `deit_tiny_baseline.yaml`
  方案A主锚点：224 输入的标准 DeiT-Tiny baseline。

- `deit_tiny_patch8_112_baseline.yaml`
  方案B主 baseline：64 -> 112, patch8。

- `deit_tiny_patch8_112_baseline_300ep.yaml`
  方案B baseline 长训版，用于和 150 epoch baseline 做纯训练时长对比。

- `deit_tiny_patch8_112_overlap_patch12.yaml`
  方案B 当前有效的结构改进：overlap patch embedding。

- `deit_tiny_patch8_112_overlap_patch12_teacher_twostage.yaml`
  单命令复现版：完整训练 + 尾训两阶段 teacher。

- `deit_tiny_patch8_112_overlap_patch12_distilled_teacher.yaml`
  teacher 自蒸馏版，供后续 student KD 对照使用。

- `deit_tiny_patch8_112_student_depth10_logit_softkd.yaml`
  depth10 student 强化配置：保持轻量化结构，训练 recipe 对齐 patch8_112 baseline，并改用 soft logit KD 提升 teacher 知识迁移效率。

## Archive Policy

- `configs/archive/` 中保留历史配置，便于回查旧实验。
- 已归档的配置主要包括：
  - 已确认收益不明显的局部模块尝试
  - 已被更新主线替代的中间配置
  - 已被 `teacher_twostage` / `logit_softkd` 替代的中间 teacher 与旧 student KD 配置
  - 早期 50 epoch / tail tuning / student 搜索配置

## Notes

- 旧的 `results/summary/*.md` 仍记录原始 config 路径；归档后如果需要严格按旧路径复现，可手动改用 `configs/archive/` 下对应文件。
