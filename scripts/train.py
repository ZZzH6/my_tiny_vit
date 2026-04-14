from __future__ import annotations

import argparse
import contextlib
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.build_loader import build_loader
from engine.evaluator import evaluate
from engine.trainer import train_one_epoch
from models.baseline_models import build_model_from_cfg
from utils.artifacts import build_run_paths, dump_csv, dump_json
from utils.model_profile import profile_model
from utils.reproducibility import seed_everything

METRIC_FIELDS = [
    "epoch",
    "lr",
    "train_loss",
    "train_acc",
    "val_acc",
    "val_top5",
    "best_acc",
    "epoch_time_sec",
]

ARTIFACT_KEYS = [
    "log_path",
    "metrics_path",
    "summary_path",
    "eval_path",
    "best_checkpoint_path",
    "last_checkpoint_path",
]


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
    parser.add_argument("--resume", default=None, help="Resume from a *_last.pt checkpoint.")
    return parser.parse_args()


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


def _epoch_lr(base_lr, epoch, total_epochs, warmup_epochs, min_lr):
    total_epochs = max(1, int(total_epochs))
    warmup_epochs = max(0, min(int(warmup_epochs), total_epochs - 1))
    base_lr = float(base_lr)
    min_lr = float(min_lr)

    if warmup_epochs > 0 and epoch < warmup_epochs:
        progress = epoch / max(1, warmup_epochs)
        return base_lr * (0.1 + 0.9 * progress)

    cosine_epochs = max(1, total_epochs - warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, cosine_epochs - 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def _cpu_state_dict(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _move_to_device(value: Any, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            state[key] = _move_to_device(value, device)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Checkpoint not found: {path}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint from {path}: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint at {path} must be a dict, got {type(checkpoint).__name__}")
    return checkpoint


def _artifact_paths_to_str(paths: dict[str, Path | str]) -> dict[str, str]:
    return {key: str(paths[key]) for key in ARTIFACT_KEYS}


def _resolve_paths(config_stem: str, resume_path: Path | None):
    results_root = ROOT / "results"
    if resume_path is None:
        return build_run_paths(results_root, config_stem), None

    checkpoint = _load_checkpoint(resume_path)
    artifact_paths = checkpoint.get("artifact_paths")
    if not isinstance(artifact_paths, dict):
        raise ValueError(f"Resume checkpoint {resume_path} is missing artifact_paths")

    missing_keys = [key for key in ARTIFACT_KEYS if key not in artifact_paths]
    if missing_keys:
        raise ValueError(f"Resume checkpoint {resume_path} is missing artifact paths: {missing_keys}")

    paths: dict[str, Path | str] = {
        "run_id": str(checkpoint.get("run_id", Path(artifact_paths["log_path"]).stem)),
        "date_str": str(checkpoint.get("date_str", str(checkpoint.get("run_id", ""))[:8])),
    }
    for key in ARTIFACT_KEYS:
        paths[key] = Path(artifact_paths[key])
    return paths, checkpoint


def _build_checkpoint_payload(
    cfg,
    paths,
    profile,
    current_epoch: int,
    best_acc: float,
    best_epoch: int,
    model_state,
):
    return {
        "run_id": str(paths["run_id"]),
        "date_str": str(paths["date_str"]),
        "config": cfg,
        "config_path": cfg.get("_config_path"),
        "artifact_paths": _artifact_paths_to_str(paths),
        "model_name": cfg["model"]["name"],
        "dataset": cfg["data"]["dataset"],
        "dataset_root": str(Path(cfg["data"]["root"]).resolve()),
        "img_size": int(cfg["data"]["img_size"]),
        "batch_size": int(cfg["data"]["batch_size"]),
        "current_epoch": int(current_epoch),
        "best_acc": float(best_acc),
        "best_epoch": int(best_epoch),
        "params": int(profile["params"]),
        "params_m": float(profile["params_m"]),
        "flops": profile["flops"],
        "flops_g": profile["flops_g"],
        "flops_note": profile["flops_note"],
        "model_state": model_state,
    }


def _render_summary_md(summary: dict[str, Any]) -> str:
    lines = [
        "# DeiT-Tiny Baseline Summary",
        "",
        "## Config",
        f"- config_path: {summary['config_path']}",
        f"- model_name: {summary['model_name']}",
        f"- dataset: {summary['dataset']}",
        f"- dataset_root: {summary['dataset_root']}",
        f"- img_size: {summary['img_size']}",
        f"- batch_size: {summary['batch_size']}",
        f"- epochs: {summary['epochs']}",
        f"- pretrained: {summary['pretrained']}",
        f"- label_smoothing: {summary['label_smoothing']}",
        "",
        "## Results",
        f"- best_epoch: {summary['best_epoch']}",
        f"- best_val_acc: {summary['best_val_acc']:.2f}",
        f"- eval_top1: {summary['eval_top1']:.2f}",
        f"- eval_top5: {summary['eval_top5']:.2f}",
        f"- total_train_time_sec: {summary['total_train_time_sec']:.2f}",
        f"- params_m: {summary['params_m']:.2f}",
        f"- flops_g: {summary['flops_g'] if summary['flops_g'] is not None else 'N/A'}",
        f"- flops_note: {summary['flops_note']}",
        "",
        "## Artifacts",
        f"- log_path: {summary['log_path']}",
        f"- metrics_path: {summary['metrics_path']}",
        f"- summary_path: {summary['summary_path']}",
        f"- eval_path: {summary['eval_path']}",
        f"- best_checkpoint_path: {summary['best_checkpoint_path']}",
        f"- last_checkpoint_path: {summary['last_checkpoint_path']}",
        "",
        "## Commands",
        f"- train: {summary['train_command']}",
        f"- eval_val: {summary['val_eval_command']}",
        f"- predict_test: {summary['test_inference_command']}",
    ]
    return "\n".join(lines) + "\n"


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main():
    args = parse_args()

    config_path = Path(args.config).resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(config_path)

    seed = int(_get(cfg, "train", "seed", default=42))
    deterministic = bool(_get(cfg, "train", "deterministic", default=True))
    seed_everything(seed, deterministic=deterministic)

    device_cfg = _get(cfg, "train", "device", default="cpu")
    device = torch.device("cuda" if device_cfg == "cuda" and torch.cuda.is_available() else "cpu")

    resume_path = Path(args.resume).expanduser().resolve() if args.resume else None
    paths, resume_checkpoint = _resolve_paths(config_path.stem, resume_path)
    for key in ARTIFACT_KEYS:
        Path(paths[key]).parent.mkdir(parents=True, exist_ok=True)

    with open(Path(paths["log_path"]), "a", encoding="utf-8") as log_file:
        tee_out = _TeeWriter(sys.__stdout__, log_file)
        tee_err = _TeeWriter(sys.__stderr__, log_file)
        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            model_cfg = cfg["model"]
            data_cfg = cfg["data"]
            train_cfg = cfg["train"]

            model_init_pretrained = bool(model_cfg["pretrained"]) and resume_checkpoint is None
            model = build_model_from_cfg(model_cfg, pretrained_override=model_init_pretrained)
            profile = profile_model(
                model,
                input_size=(3, int(data_cfg["img_size"]), int(data_cfg["img_size"])),
            )
            model = model.to(device)

            train_loader, val_loader = build_loader(cfg)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(train_cfg["lr"]),
                weight_decay=float(train_cfg["weight_decay"]),
            )
            criterion = torch.nn.CrossEntropyLoss(
                label_smoothing=float(_get(cfg, "train", "label_smoothing", default=0.0))
            )
            amp_enabled = bool(_get(cfg, "train", "amp", default=True)) and device.type == "cuda"
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
            max_grad_norm = float(_get(cfg, "train", "max_grad_norm", default=1.0))

            history: list[dict[str, Any]] = []
            start_epoch = 0
            best_acc = float("-inf")
            best_epoch = 0
            total_train_time_sec = 0.0

            if resume_checkpoint is not None:
                model.load_state_dict(resume_checkpoint["model_state"])
                if "optimizer_state" in resume_checkpoint:
                    optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
                    _move_optimizer_state_to_device(optimizer, device)
                if "scaler_state" in resume_checkpoint:
                    scaler.load_state_dict(resume_checkpoint["scaler_state"])
                history = list(resume_checkpoint.get("history", []))
                start_epoch = int(resume_checkpoint.get("current_epoch", 0))
                best_acc = float(resume_checkpoint.get("best_acc", float("-inf")))
                best_epoch = int(resume_checkpoint.get("best_epoch", 0))
                total_train_time_sec = float(resume_checkpoint.get("total_train_time_sec", 0.0))

            dump_csv(Path(paths["metrics_path"]), history, METRIC_FIELDS)

            print("=" * 80)
            print("Tiny-ImageNet | DeiT-Tiny baseline training")
            print("=" * 80)
            print(f"config     : {cfg['_config_path']}")
            print(f"device     : {device}")
            print(f"dataset    : {data_cfg['dataset']}")
            print(f"data root  : {data_cfg['root']}")
            print(f"img size   : {data_cfg['img_size']}")
            print(f"batch size : {data_cfg['batch_size']}")
            print(f"epochs     : {train_cfg['epochs']}")
            print(f"lr         : {train_cfg['lr']}")
            print(f"wd         : {train_cfg['weight_decay']}")
            print(f"pretrained : {model_init_pretrained}")
            print(f"resume     : {resume_path if resume_path is not None else 'N/A'}")
            print(f"Params     : {profile['params_m']:.2f}M")
            if profile["flops_g"] is not None:
                print(f"FLOPs      : {profile['flops_g']:.2f}G")
            else:
                print(f"FLOPs      : N/A ({profile['flops_note']})")
            print("-" * 80)
            print(
                f"{'epoch':>5} | {'lr':>10} | {'train_loss':>10} | {'train_acc(%)':>12} | "
                f"{'val_acc(%)':>10} | {'best_acc(%)':>10} | {'time':>8}"
            )
            print("-" * 80)

            epochs = int(train_cfg["epochs"])
            warmup_epochs = int(_get(cfg, "train", "warmup_epochs", default=5))
            min_lr = float(_get(cfg, "train", "min_lr", default=1.0e-6))
            base_lr = float(train_cfg["lr"])

            for epoch_idx in range(start_epoch, epochs):
                epoch_number = epoch_idx + 1
                lr_now = _epoch_lr(base_lr, epoch_idx, epochs, warmup_epochs, min_lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_now

                start_time = time.perf_counter()
                train_metrics = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    criterion,
                    device,
                    scaler=scaler,
                    max_grad_norm=max_grad_norm,
                )
                eval_metrics = evaluate(model, val_loader, device)
                epoch_time = time.perf_counter() - start_time
                total_train_time_sec += epoch_time

                train_loss = float(train_metrics["loss"])
                train_acc = float(train_metrics["acc"])
                val_acc = float(eval_metrics["top1"])
                val_top5 = float(eval_metrics["top5"])
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_epoch = epoch_number
                    best_payload = _build_checkpoint_payload(
                        cfg,
                        paths,
                        profile,
                        epoch_number,
                        best_acc,
                        best_epoch,
                        _cpu_state_dict(model),
                    )
                    best_payload["type"] = "best"
                    _save_checkpoint(Path(paths["best_checkpoint_path"]), best_payload)

                history.append(
                    {
                        "epoch": epoch_number,
                        "lr": lr_now,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_acc": val_acc,
                        "val_top5": val_top5,
                        "best_acc": best_acc,
                        "epoch_time_sec": epoch_time,
                    }
                )
                dump_csv(Path(paths["metrics_path"]), history, METRIC_FIELDS)

                last_payload = _build_checkpoint_payload(
                    cfg,
                    paths,
                    profile,
                    epoch_number,
                    best_acc,
                    best_epoch,
                    _cpu_state_dict(model),
                )
                last_payload.update(
                    {
                        "type": "last",
                        "optimizer_state": optimizer.state_dict(),
                        "scaler_state": scaler.state_dict(),
                        "history": history,
                        "total_train_time_sec": float(total_train_time_sec),
                    }
                )
                _save_checkpoint(Path(paths["last_checkpoint_path"]), last_payload)

                print(
                    f"{epoch_number:5d} | {lr_now:10.6f} | {train_loss:10.4f} | "
                    f"{train_acc:12.2f} | {val_acc:10.2f} | {best_acc:10.2f} | {epoch_time:8.1f}s"
                )

            best_checkpoint = _load_checkpoint(Path(paths["best_checkpoint_path"]))
            model.load_state_dict(best_checkpoint["model_state"])
            eval_metrics = evaluate(model, val_loader, device)
            eval_result = {
                "model_name": model_cfg["name"],
                "checkpoint_path": str(paths["best_checkpoint_path"]),
                "dataset_root": str(Path(data_cfg["root"]).resolve()),
                "split": "val",
                "top1": float(eval_metrics["top1"]),
                "top5": float(eval_metrics["top5"]),
                "num_samples": int(eval_metrics["num_samples"]),
                "img_size": int(data_cfg["img_size"]),
                "batch_size": int(data_cfg["batch_size"]),
                "device": str(device),
            }
            dump_json(Path(paths["eval_path"]), eval_result)

            summary = {
                "config_path": cfg["_config_path"],
                "model_name": model_cfg["name"],
                "dataset": data_cfg["dataset"],
                "dataset_root": str(Path(data_cfg["root"]).resolve()),
                "img_size": int(data_cfg["img_size"]),
                "batch_size": int(data_cfg["batch_size"]),
                "epochs": epochs,
                "pretrained": bool(model_cfg["pretrained"]),
                "label_smoothing": float(_get(cfg, "train", "label_smoothing", default=0.0)),
                "best_epoch": best_epoch,
                "best_val_acc": float(best_acc),
                "eval_top1": float(eval_result["top1"]),
                "eval_top5": float(eval_result["top5"]),
                "total_train_time_sec": float(total_train_time_sec),
                "params_m": float(profile["params_m"]),
                "flops_g": profile["flops_g"],
                "flops_note": profile["flops_note"],
                "log_path": str(paths["log_path"]),
                "metrics_path": str(paths["metrics_path"]),
                "summary_path": str(paths["summary_path"]),
                "eval_path": str(paths["eval_path"]),
                "best_checkpoint_path": str(paths["best_checkpoint_path"]),
                "last_checkpoint_path": str(paths["last_checkpoint_path"]),
                "train_command": f"python -u scripts/train.py --config {args.config}",
                "val_eval_command": (
                    f"python -u scripts/test.py --config {args.config} "
                    f"--checkpoint {paths['best_checkpoint_path']} --split val"
                ),
                "test_inference_command": (
                    f"python -u scripts/test.py --config {args.config} "
                    f"--checkpoint {paths['best_checkpoint_path']} --split test"
                ),
            }
            Path(paths["summary_path"]).write_text(_render_summary_md(summary), encoding="utf-8")

            print("-" * 80)
            print(f"finished. best val acc: {best_acc:.2f}% at epoch {best_epoch}")
            print(f"best checkpoint : {paths['best_checkpoint_path']}")
            print(f"last checkpoint : {paths['last_checkpoint_path']}")
            print(f"metrics         : {paths['metrics_path']}")
            print(f"summary         : {paths['summary_path']}")
            print(f"eval result     : {paths['eval_path']}")


if __name__ == "__main__":
    main()
