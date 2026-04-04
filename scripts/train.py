from __future__ import annotations

import argparse
import contextlib
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.build_loader import build_loader
from engine.evaluator import evaluate
from engine.trainer import train_one_epoch
from models.baseline_models import build_model
from utils.reproducibility import seed_everything


def _get(cfg, *keys, default=None):
    cur = cfg
    for key in keys:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def _print_header(cfg, device, log_path):
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    weight_decay = _get(cfg, "train", "weight_decay", default="n/a")
    mixup_alpha = _get(cfg, "train", "mixup_alpha", default="n/a")
    cutmix_alpha = _get(cfg, "train", "cutmix_alpha", default="n/a")
    label_smoothing = _get(cfg, "train", "label_smoothing", default="n/a")
    drop_path_rate = _get(cfg, "model", "drop_path_rate", default="n/a")
    warmup_epochs = _get(cfg, "train", "warmup_epochs", default="n/a")
    early_stop_patience = _get(cfg, "train", "early_stop_patience", default="n/a")
    seed = _get(cfg, "train", "seed", default="n/a")
    deterministic = _get(cfg, "train", "deterministic", default="n/a")

    print("=" * 72)
    print("Tiny-ImageNet-200 | DeiT-Tiny baseline training")
    print("=" * 72)
    print(f"log file   : {log_path}")
    print(f"model      : {model_cfg['name']}  (pretrained={model_cfg['pretrained']})")
    print(f"classes    : {model_cfg['num_classes']}")
    print(f"data root  : {data_cfg['root']}")
    print(f"batch size : {data_cfg['batch_size']}")
    print(f"epochs     : {train_cfg['epochs']}")
    print(f"lr         : {train_cfg['lr']}")
    print(f"wd         : {weight_decay}")
    print(f"mixup      : {mixup_alpha} / cutmix {cutmix_alpha}")
    print(f"label_smooth: {label_smoothing}")
    print(f"drop_path  : {drop_path_rate}")
    print(f"warmup     : {warmup_epochs} epochs")
    print(f"patience   : {early_stop_patience} epochs")
    print(f"seed       : {seed} (deterministic={deterministic})")
    print(f"device     : {device}")
    print("-" * 72)
    print(f"{'epoch':>5} | {'train_loss':>10} | {'val_acc(%)':>10} | {'best_acc(%)':>10} | {'time':>8}")
    print("-" * 72)


class _TeeWriter:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _epoch_lr(base_lr, epoch, total_epochs, warmup_epochs, min_lr, warmup_start_factor=0.1):
    total_epochs = max(1, int(total_epochs))
    warmup_epochs = max(0, min(int(warmup_epochs), total_epochs - 1))
    base_lr = float(base_lr)
    min_lr = float(min_lr)

    if warmup_epochs > 0 and epoch < warmup_epochs:
        progress = epoch / max(1, warmup_epochs - 1)
        return base_lr * (warmup_start_factor + (1.0 - warmup_start_factor) * progress)

    cosine_epochs = max(1, total_epochs - warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, cosine_epochs - 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def _build_criterion(cfg, use_mixup):
    label_smoothing = float(_get(cfg, "train", "label_smoothing", default=0.1))
    if use_mixup:
        return SoftTargetCrossEntropy()
    if label_smoothing > 0.0:
        return LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    return torch.nn.CrossEntropyLoss()


def _cpu_state_dict(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(_get(cfg, "train", "seed", default=42))
    deterministic = bool(_get(cfg, "train", "deterministic", default=True))
    seed_everything(seed, deterministic=deterministic)

    device_cfg = _get(cfg, "train", "device", default="cpu")
    if device_cfg == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%Y%m%d_%H%M%S")
    config_stem = Path(args.config).stem

    log_dir = ROOT / "results" / "logs" / date_str
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{config_stem}_{time_str}.log"
    ckpt_dir = ROOT / "results" / "checkpoints" / date_str
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{config_stem}_{time_str}_best.pt"

    with open(log_path, "a", encoding="utf-8") as log_file:
        tee_out = _TeeWriter(sys.__stdout__, log_file)
        tee_err = _TeeWriter(sys.__stderr__, log_file)
        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            print(f"log file   : {log_path}")

            model_cfg = cfg["model"]
            model = build_model(
                model_name=model_cfg["name"],
                num_classes=int(model_cfg["num_classes"]),
                pretrained=bool(model_cfg["pretrained"]),
                drop_path_rate=float(_get(cfg, "model", "drop_path_rate", default=0.1)),
                drop_rate=float(_get(cfg, "model", "drop_rate", default=0.0)),
                attn_drop_rate=float(_get(cfg, "model", "attn_drop_rate", default=0.0)),
            ).to(device)

            train_loader, val_loader, mixup_fn = build_loader(cfg)
            train_cfg = cfg["train"]
            base_lr = float(train_cfg["lr"])
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=base_lr,
                weight_decay=float(_get(cfg, "train", "weight_decay", default=0.05)),
            )
            criterion = _build_criterion(cfg, use_mixup=mixup_fn is not None)
            amp_enabled = bool(_get(cfg, "train", "amp", default=True)) and device.type == "cuda"
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
            max_grad_norm = float(_get(cfg, "train", "max_grad_norm", default=1.0))

            _print_header(cfg, device, log_path)

            epochs = int(cfg["train"]["epochs"])
            warmup_epochs = int(_get(cfg, "train", "warmup_epochs", default=5))
            min_lr = float(_get(cfg, "train", "min_lr", default=1e-6))
            early_stop_patience = int(_get(cfg, "train", "early_stop_patience", default=10))
            min_delta = float(_get(cfg, "train", "min_delta", default=0.0))

            best_acc = float("-inf")
            best_epoch = 0
            best_state = _cpu_state_dict(model)
            stale_epochs = 0

            for epoch in range(epochs):
                lr_now = _epoch_lr(base_lr, epoch, epochs, warmup_epochs, min_lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_now

                start = time.perf_counter()
                loss = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    criterion,
                    device,
                    mixup_fn=mixup_fn,
                    scaler=scaler,
                    max_grad_norm=max_grad_norm,
                )
                acc = evaluate(model, val_loader, device)
                if acc > best_acc + min_delta:
                    best_acc = acc
                    best_epoch = epoch + 1
                    best_state = _cpu_state_dict(model)
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                elapsed = time.perf_counter() - start
                print(
                    f"{epoch + 1:5d} | {loss:10.4f} | {acc:10.2f} | {best_acc:10.2f} | {elapsed:8.1f}s"
                )
                if early_stop_patience > 0 and stale_epochs >= early_stop_patience:
                    print(
                        f"early stop triggered at epoch {epoch + 1} "
                        f"after {stale_epochs} stale epochs."
                    )
                    break

            model.load_state_dict(best_state)
            torch.save(
                {
                    "model_state": best_state,
                    "best_acc": best_acc,
                    "best_epoch": best_epoch,
                    "config": cfg,
                },
                ckpt_path,
            )

            print("-" * 72)
            print(f"finished. best val acc: {best_acc:.2f}% at epoch {best_epoch}")
            print(f"best checkpoint saved to: {ckpt_path}")
            print(f"log saved to: {log_path}")


if __name__ == "__main__":
    main()
