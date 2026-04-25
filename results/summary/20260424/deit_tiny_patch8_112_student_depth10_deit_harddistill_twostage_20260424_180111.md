# DeiT-Tiny Experiment Summary

## Config
- config_path: /home/zjhao/bishe/my_tiny_vit/configs/deit_tiny_patch8_112_student_depth10_deit_harddistill_twostage.yaml
- model_name: deit_tiny_patch8_112
- dataset: tiny_imagenet
- dataset_root: /home/zjhao/bishe/my_tiny_vit/dataset/tiny-imagenet-200
- img_size: 112
- batch_size: 256
- epochs: 190
- pretrained: True
- init_checkpoint: N/A
- distillation_enabled: True
- distillation_method: deit
- distillation_type: hard
- teacher_checkpoint: /home/zjhao/bishe/my_tiny_vit/results/checkpoints/20260422/deit_tiny_patch8_112_overlap_patch12_strong_teacher_polish40_20260422_171852_best.pt
- distillation_alpha: 0.4
- distillation_temperature: 1.0
- label_smoothing: 0.1

## Results
- best_epoch: 179
- best_model_source: ema
- best_val_acc: 79.41
- eval_top1: 79.41
- eval_top5: 93.28
- total_train_time_sec: 12349.81
- params_m: 4.60
- flops_g: 1.766381568
- flops_note: estimated with thop; FLOPs=2xMACs

## Artifacts
- log_path: /home/zjhao/bishe/my_tiny_vit/results/logs/20260424/deit_tiny_patch8_112_student_depth10_deit_harddistill_twostage_20260424_180111.log
- metrics_path: /home/zjhao/bishe/my_tiny_vit/results/metrics/20260424/deit_tiny_patch8_112_student_depth10_deit_harddistill_twostage_20260424_180111.csv
- summary_path: /home/zjhao/bishe/my_tiny_vit/results/summary/20260424/deit_tiny_patch8_112_student_depth10_deit_harddistill_twostage_20260424_180111.md
- eval_path: /home/zjhao/bishe/my_tiny_vit/results/eval/20260424/deit_tiny_patch8_112_student_depth10_deit_harddistill_twostage_20260424_180111_val.json
- best_checkpoint_path: /home/zjhao/bishe/my_tiny_vit/results/checkpoints/20260424/deit_tiny_patch8_112_student_depth10_deit_harddistill_twostage_20260424_180111_best.pt
- last_checkpoint_path: /home/zjhao/bishe/my_tiny_vit/results/checkpoints/20260424/deit_tiny_patch8_112_student_depth10_deit_harddistill_twostage_20260424_180111_last.pt

## Commands
- train: python -u scripts/train.py --config configs/deit_tiny_patch8_112_student_depth10_deit_harddistill_twostage.yaml
- eval_val: python -u scripts/test.py --config configs/deit_tiny_patch8_112_student_depth10_deit_harddistill_twostage.yaml --checkpoint /home/zjhao/bishe/my_tiny_vit/results/checkpoints/20260424/deit_tiny_patch8_112_student_depth10_deit_harddistill_twostage_20260424_180111_best.pt --split val
- predict_test: python -u scripts/test.py --config configs/deit_tiny_patch8_112_student_depth10_deit_harddistill_twostage.yaml --checkpoint /home/zjhao/bishe/my_tiny_vit/results/checkpoints/20260424/deit_tiny_patch8_112_student_depth10_deit_harddistill_twostage_20260424_180111_best.pt --split test
