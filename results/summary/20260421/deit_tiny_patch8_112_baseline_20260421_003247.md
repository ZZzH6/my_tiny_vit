# DeiT-Tiny Experiment Summary

## Config
- config_path: /home/zjhao/bishe/my_tiny_vit/configs/deit_tiny_patch8_112_baseline.yaml
- model_name: deit_tiny_patch8_112
- dataset: tiny_imagenet
- dataset_root: /home/zjhao/bishe/my_tiny_vit/dataset/tiny-imagenet-200
- img_size: 112
- batch_size: 256
- epochs: 150
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
- best_epoch: 146
- best_model_source: ema
- best_val_acc: 79.46
- eval_top1: 79.46
- eval_top5: 93.18
- total_train_time_sec: 15736.20
- params_m: 5.45
- flops_g: 2.106043392
- flops_note: estimated with thop; FLOPs=2xMACs

## Artifacts
- log_path: /home/zjhao/bishe/my_tiny_vit/results/logs/20260421/deit_tiny_patch8_112_baseline_20260421_003247.log
- metrics_path: /home/zjhao/bishe/my_tiny_vit/results/metrics/20260421/deit_tiny_patch8_112_baseline_20260421_003247.csv
- summary_path: /home/zjhao/bishe/my_tiny_vit/results/summary/20260421/deit_tiny_patch8_112_baseline_20260421_003247.md
- eval_path: /home/zjhao/bishe/my_tiny_vit/results/eval/20260421/deit_tiny_patch8_112_baseline_20260421_003247_val.json
- best_checkpoint_path: /home/zjhao/bishe/my_tiny_vit/results/checkpoints/20260421/deit_tiny_patch8_112_baseline_20260421_003247_best.pt
- last_checkpoint_path: /home/zjhao/bishe/my_tiny_vit/results/checkpoints/20260421/deit_tiny_patch8_112_baseline_20260421_003247_last.pt

## Commands
- train: python -u scripts/train.py --config configs/deit_tiny_patch8_112_baseline.yaml
- eval_val: python -u scripts/test.py --config configs/deit_tiny_patch8_112_baseline.yaml --checkpoint /home/zjhao/bishe/my_tiny_vit/results/checkpoints/20260421/deit_tiny_patch8_112_baseline_20260421_003247_best.pt --split val
- predict_test: python -u scripts/test.py --config configs/deit_tiny_patch8_112_baseline.yaml --checkpoint /home/zjhao/bishe/my_tiny_vit/results/checkpoints/20260421/deit_tiny_patch8_112_baseline_20260421_003247_best.pt --split test
