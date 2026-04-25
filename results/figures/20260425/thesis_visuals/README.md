# Thesis Visual Outputs

本目录包含可直接用于毕业论文的高分辨率曲线图、对比表和合并 PDF。

## 关于 train loss

- 本次默认不单独输出 train loss 图。
- 原因是当前主线训练广泛使用了 mixup、cutmix、label smoothing 与 EMA。
- 在这种设置下，train loss 不再对应传统 one-hot 监督下的“是否过拟合”，数值也不会像常规训练那样下降到很低。
- 因此论文主体更建议使用 val acc 曲线来展示收敛速度、稳定性和模型间差异。

## 输出文件

- `curve_mainline_overview.pdf`
- `curve_mainline_overview.png`
- `curve_structure_ablation_112.pdf`
- `curve_structure_ablation_112.png`
- `curve_student_selection.pdf`
- `curve_student_selection.png`
- `curve_teacher_evolution.pdf`
- `curve_teacher_evolution.png`
- `table_main_results.csv`
- `table_main_results.pdf`
- `table_main_results.png`
- `table_structure_ablation_112.csv`
- `table_structure_ablation_112.pdf`
- `table_structure_ablation_112.png`
- `table_student_selection.csv`
- `table_student_selection.pdf`
- `table_student_selection.png`
- `table_teacher_evolution.csv`
- `table_teacher_evolution.pdf`
- `table_teacher_evolution.png`
- `thesis_visuals_bundle.pdf`
