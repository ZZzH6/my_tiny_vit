# DeiT-Tiny Experiment Summary

## Config
- config_path: /home/zjhao/bishe/my_tiny_vit/configs/deit_tiny_patch8_112_overlap_patch12_teacher_twostage.yaml
- model_name: deit_tiny_patch8_112
- dataset: tiny_imagenet
- dataset_root: /home/zjhao/bishe/my_tiny_vit/dataset/tiny-imagenet-200
- img_size: 112
- batch_size: 256
- epochs: 190
- pretrained: True
- init_checkpoint: N/A
- distillation_enabled: False
- distillation_method: N/A
- distillation_type: N/A
- teacher_checkpoint: N/A
- distillation_alpha: 0.0
- distillation_temperature: 1.0
- label_smoothing: 0.1

## Results
- best_epoch: 189
- best_model_source: ema
- best_val_acc: 80.18
- eval_top1: 80.18
- eval_top5: 94.07
- total_train_time_sec: 11463.11
- params_m: 5.50
- flops_g: 2.124106752
- flops_note: estimated with thop; FLOPs=2xMACs

## Artifacts
- log_path: /home/zjhao/bishe/my_tiny_vit/results/logs/20260423/deit_tiny_patch8_112_overlap_patch12_teacher_twostage_20260423_232954.log
- metrics_path: /home/zjhao/bishe/my_tiny_vit/results/metrics/20260423/deit_tiny_patch8_112_overlap_patch12_teacher_twostage_20260423_232954.csv
- summary_path: /home/zjhao/bishe/my_tiny_vit/results/summary/20260423/deit_tiny_patch8_112_overlap_patch12_teacher_twostage_20260423_232954.md
- eval_path: /home/zjhao/bishe/my_tiny_vit/results/eval/20260423/deit_tiny_patch8_112_overlap_patch12_teacher_twostage_20260423_232954_val.json
- best_checkpoint_path: /home/zjhao/bishe/my_tiny_vit/results/checkpoints/20260423/deit_tiny_patch8_112_overlap_patch12_teacher_twostage_20260423_232954_best.pt
- last_checkpoint_path: /home/zjhao/bishe/my_tiny_vit/results/checkpoints/20260423/deit_tiny_patch8_112_overlap_patch12_teacher_twostage_20260423_232954_last.pt

## Commands
- train: python -u scripts/train.py --config configs/deit_tiny_patch8_112_overlap_patch12_teacher_twostage.yaml
- eval_val: python -u scripts/test.py --config configs/deit_tiny_patch8_112_overlap_patch12_teacher_twostage.yaml --checkpoint /home/zjhao/bishe/my_tiny_vit/results/checkpoints/20260423/deit_tiny_patch8_112_overlap_patch12_teacher_twostage_20260423_232954_best.pt --split val
- predict_test: python -u scripts/test.py --config configs/deit_tiny_patch8_112_overlap_patch12_teacher_twostage.yaml --checkpoint /home/zjhao/bishe/my_tiny_vit/results/checkpoints/20260423/deit_tiny_patch8_112_overlap_patch12_teacher_twostage_20260423_232954_best.pt --split test
