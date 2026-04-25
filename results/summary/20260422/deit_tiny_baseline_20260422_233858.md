# DeiT-Tiny Experiment Summary

## Config
- config_path: /home/zjhao/bishe/my_tiny_vit/configs/deit_tiny_baseline.yaml
- model_name: deit_tiny
- dataset: tiny_imagenet
- dataset_root: /home/zjhao/bishe/my_tiny_vit/dataset/tiny-imagenet-200
- img_size: 224
- batch_size: 512
- epochs: 300
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
- best_epoch: 299
- best_model_source: ema
- best_val_acc: 77.37
- eval_top1: 77.37
- eval_top5: 92.59
- total_train_time_sec: 19429.72
- params_m: 5.56
- flops_g: 2.149395456
- flops_note: estimated with thop; FLOPs=2xMACs

## Artifacts
- log_path: /home/zjhao/bishe/my_tiny_vit/results/logs/20260422/deit_tiny_baseline_20260422_233858.log
- metrics_path: /home/zjhao/bishe/my_tiny_vit/results/metrics/20260422/deit_tiny_baseline_20260422_233858.csv
- summary_path: /home/zjhao/bishe/my_tiny_vit/results/summary/20260422/deit_tiny_baseline_20260422_233858.md
- eval_path: /home/zjhao/bishe/my_tiny_vit/results/eval/20260422/deit_tiny_baseline_20260422_233858_val.json
- best_checkpoint_path: /home/zjhao/bishe/my_tiny_vit/results/checkpoints/20260422/deit_tiny_baseline_20260422_233858_best.pt
- last_checkpoint_path: /home/zjhao/bishe/my_tiny_vit/results/checkpoints/20260422/deit_tiny_baseline_20260422_233858_last.pt

## Commands
- train: python -u scripts/train.py --config configs/deit_tiny_baseline.yaml
- eval_val: python -u scripts/test.py --config configs/deit_tiny_baseline.yaml --checkpoint /home/zjhao/bishe/my_tiny_vit/results/checkpoints/20260422/deit_tiny_baseline_20260422_233858_best.pt --split val
- predict_test: python -u scripts/test.py --config configs/deit_tiny_baseline.yaml --checkpoint /home/zjhao/bishe/my_tiny_vit/results/checkpoints/20260422/deit_tiny_baseline_20260422_233858_best.pt --split test
