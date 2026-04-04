from __future__ import annotations

import argparse
import contextlib
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

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
from utils.artifacts import build_run_paths, dump_csv, dump_json
from utils.model_profile import profile_model
from utils.reproducibility import seed_everything

METRIC_FIELDS = [
    "epoch",
    "lr",
    "train_loss",
    "val_acc",
    "val_top5",
    "best_acc",
    "epoch_time_sec",
]


def _get(cfg, *keys, default=None):
    cur = cfg
    for key in keys:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def _format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _format_path(path: Path | None) -> str:
    return str(path) if path is not None else "N/A"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="Resume from a *_last.pt checkpoint.")
    return parser.parse_args()


def _print_header(cfg, device, paths, profile, resume_from):
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    print("=" * 80)
    print("Tiny-ImageNet-200 | DeiT-Tiny formal baseline training")
    print("=" * 80)
    print(f"config     : {cfg.get('_config_path', 'N/A')}")
    print(f"log file   : {paths['log_path']}")
    print(f"metrics    : {paths['metrics_path']}")
    print(f"summary    : {paths['summary_path']}")
    print(f"best ckpt  : {paths['best_checkpoint_path']}")
    print(f"last ckpt  : {paths['last_checkpoint_path']}")
    print(f"eval file  : {paths['eval_path']}")
    print(f"resume from: {resume_from if resume_from is not None else 'N/A'}")
    print(f"model      : {model_cfg['name']}  (pretrained={model_cfg['pretrained']})")
    print(f"dataset    : {data_cfg['dataset']}")
    print(f"data root  : {data_cfg['root']}")
    print(f"img size   : {data_cfg['img_size']}")
    print(f"batch size : {data_cfg['batch_size']}")
    print(f"epochs     : {train_cfg['epochs']}")
    print(f"lr         : {train_cfg['lr']}")
    print(f"wd         : {train_cfg['weight_decay']}")
    print(f"mixup      : {train_cfg['mixup_alpha']} / cutmix {train_cfg['cutmix_alpha']}")
    print(f"label_smooth: {train_cfg['label_smoothing']}")
    print(f"drop_path  : {model_cfg['drop_path_rate']}")
    print(f"warmup     : {train_cfg['warmup_epochs']} epochs")
    print(f"patience   : {train_cfg['early_stop_patience']} epochs")
    print(f"seed       : {train_cfg['seed']} (deterministic={train_cfg['deterministic']})")
    print(f"Params     : {profile['params_m']:.2f}M")
    if profile["flops_g"] is not None:
        print(f"FLOPs      : {profile['flops_g']:.2f}G ({profile['flops_note']})")
    else:
        print(f"FLOPs      : N/A ({profile['flops_note']})")
    print(f"device     : {device}")
    print("-" * 80)
    print(f"{'epoch':>5} | {'lr':>10} | {'train_loss':>10} | {'val_acc(%)':>10} | {'best_acc(%)':>10} | {'time':>8}")
    print("-" * 80)


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
    except Exception as exc:  # pragma: no cover - defensive error wrapping
        raise RuntimeError(f"Failed to load checkpoint from {path}: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint at {path} must be a dict, got {type(checkpoint).__name__}")
    return checkpoint


def _require_keys(container: dict[str, Any], keys: list[str], label: str) -> None:
    missing = [key for key in keys if key not in container]
    if missing:
        raise ValueError(f"{label} is missing required keys: {missing}")


def _check_resume_compatibility(cfg, checkpoint_cfg):
    required_pairs = [
        ("model.name", ("model", "name")),
        ("model.num_classes", ("model", "num_classes")),
        ("model.pretrained", ("model", "pretrained")),
        ("data.dataset", ("data", "dataset")),
        ("data.img_size", ("data", "img_size")),
        ("train.seed", ("train", "seed")),
        ("train.deterministic", ("train", "deterministic")),
    ]
    for label, cfg_keys in required_pairs:
        current_value = _get(cfg, *cfg_keys)
        checkpoint_value = _get(checkpoint_cfg, *cfg_keys)
        if current_value != checkpoint_value:
            raise ValueError(
                f"Resume checkpoint is incompatible for {label}: current={current_value!r}, "
                f"checkpoint={checkpoint_value!r}"
            )


def _resolve_paths(cfg, args):
    config_stem = Path(args.config).stem
    results_root = ROOT / "results"
    resume_path = Path(args.resume).expanduser() if args.resume else None

    if resume_path is None:
        paths = build_run_paths(results_root, config_stem, eval_split="val")
        return paths, None

    checkpoint = _load_checkpoint(resume_path)
    _require_keys(
        checkpoint,
        [
            "model_state",
            "optimizer_state",
            "scaler_state",
            "current_epoch",
            "best_acc",
            "best_epoch",
            "history",
            "total_train_time_sec",
            "artifact_paths",
            "config",
            "run_id",
        ],
        f"resume checkpoint {resume_path}",
    )
    if not isinstance(checkpoint["artifact_paths"], dict):
        raise ValueError(f"resume checkpoint {resume_path} has invalid artifact_paths")
    _check_resume_compatibility(cfg, checkpoint["config"])

    artifact_paths = checkpoint["artifact_paths"]
    artifact_keys = [
        "log_path",
        "metrics_path",
        "summary_path",
        "eval_path",
        "best_checkpoint_path",
        "last_checkpoint_path",
    ]
    missing_paths = [key for key in artifact_keys if key not in artifact_paths]
    if missing_paths:
        raise ValueError(f"resume checkpoint {resume_path} is missing artifact path keys: {missing_paths}")

    paths = {
        "run_id": str(checkpoint["run_id"]),
        "date_str": str(checkpoint.get("date_str", str(checkpoint["run_id"])[:8])),
        "log_path": Path(artifact_paths["log_path"]),
        "metrics_path": Path(artifact_paths["metrics_path"]),
        "summary_path": Path(artifact_paths["summary_path"]),
        "eval_path": Path(artifact_paths["eval_path"]),
        "best_checkpoint_path": Path(artifact_paths["best_checkpoint_path"]),
        "last_checkpoint_path": Path(artifact_paths["last_checkpoint_path"]),
    }
    return paths, checkpoint


def _build_common_payload(
    cfg,
    paths,
    profile,
    seed,
    deterministic,
    resume_from,
    eval_split="val",
):
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    return {
        "run_id": paths["run_id"],
        "date_str": paths["date_str"],
        "config": cfg,
        "config_path": cfg.get("_config_path"),
        "artifact_paths": {key: str(value) for key, value in paths.items() if key in {
            "log_path",
            "metrics_path",
            "summary_path",
            "eval_path",
            "best_checkpoint_path",
            "last_checkpoint_path",
        }},
        "model_name": model_cfg["name"],
        "dataset": data_cfg["dataset"],
        "dataset_root": str(Path(data_cfg["root"]).resolve()),
        "img_size": int(data_cfg["img_size"]),
        "batch_size": int(data_cfg["batch_size"]),
        "epochs_configured": int(train_cfg["epochs"]),
        "seed": int(seed),
        "deterministic": bool(deterministic),
        "pretrained": bool(model_cfg["pretrained"]),
        "mixup_alpha": float(train_cfg["mixup_alpha"]),
        "cutmix_alpha": float(train_cfg["cutmix_alpha"]),
        "label_smoothing": float(train_cfg["label_smoothing"]),
        "weight_decay": float(train_cfg["weight_decay"]),
        "drop_path_rate": float(model_cfg["drop_path_rate"]),
        "params": int(profile["params"]),
        "params_m": float(profile["params_m"]),
        "flops": profile["flops"],
        "flops_g": profile["flops_g"],
        "flops_status": profile["flops_status"],
        "flops_note": profile["flops_note"],
        "eval_split": eval_split,
        "resume_from": str(resume_from) if resume_from is not None else None,
    }


def _render_summary_md(summary: dict[str, Any]) -> str:
    def fmt(value: Any) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    lines = []
    lines.append(f"# {summary['model_name']} Baseline Summary")
    lines.append("")
    lines.append("## Config")
    lines.append("| Key | Value |")
    lines.append("| --- | --- |")
    for key in [
        "dataset",
        "dataset_root",
        "img_size",
        "batch_size",
        "epochs_configured",
        "epochs_completed",
        "seed",
        "deterministic",
        "pretrained",
        "mixup",
        "cutmix",
        "label_smoothing",
        "weight_decay",
        "drop_path_rate",
        "resume_from",
    ]:
        lines.append(f"| {key} | {fmt(summary.get(key))} |")
    lines.append("")
    lines.append("## Results")
    lines.append("| Key | Value |")
    lines.append("| --- | --- |")
    for key in [
        "best_epoch",
        "best_val_acc",
        "eval_top1",
        "eval_top5",
        "total_train_time_sec",
        "Params (M)",
        "FLOPs (G)",
        "FLOPs note",
    ]:
        lines.append(f"| {key} | {fmt(summary.get(key))} |")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("| Key | Value |")
    lines.append("| --- | --- |")
    for key in [
        "best_checkpoint_path",
        "last_checkpoint_path",
        "log_path",
        "metrics_path",
        "summary_path",
        "eval_path",
    ]:
        lines.append(f"| {key} | {fmt(summary.get(key))} |")
    lines.append("")
    lines.append("## Eval Command")
    lines.append("```bash")
    lines.append(summary["eval_command"])
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(Path(args.config).resolve())

    seed = int(_get(cfg, "train", "seed", default=42))
    deterministic = bool(_get(cfg, "train", "deterministic", default=True))
    seed_everything(seed, deterministic=deterministic)

    device_cfg = _get(cfg, "train", "device", default="cpu")
    if device_cfg == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    paths, resume_checkpoint = _resolve_paths(cfg, args)
    for key in [
        "log_path",
        "metrics_path",
        "summary_path",
        "eval_path",
        "best_checkpoint_path",
        "last_checkpoint_path",
    ]:
        paths[key].parent.mkdir(parents=True, exist_ok=True)

    with open(paths["log_path"], "a", encoding="utf-8") as log_file:
        tee_out = _TeeWriter(sys.__stdout__, log_file)
        tee_err = _TeeWriter(sys.__stderr__, log_file)
        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            print(f"log file   : {paths['log_path']}")

            model_cfg = cfg["model"]
            data_cfg = cfg["data"]
            model = build_model(
                model_name=model_cfg["name"],
                num_classes=int(model_cfg["num_classes"]),
                pretrained=bool(model_cfg["pretrained"]),
                drop_path_rate=float(_get(cfg, "model", "drop_path_rate", default=0.1)),
                drop_rate=float(_get(cfg, "model", "drop_rate", default=0.0)),
                attn_drop_rate=float(_get(cfg, "model", "attn_drop_rate", default=0.0)),
            )
            profile = profile_model(model, input_size=(3, int(data_cfg["img_size"]), int(data_cfg["img_size"])))
            model = model.to(device)

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

            history: list[dict[str, Any]] = []
            current_epoch = 0
            best_acc = float("-inf")
            best_epoch = 0
            total_train_time_sec = 0.0
            resume_from = None
            if resume_checkpoint is not None:
                resume_from = Path(args.resume).expanduser()
                try:
                    model.load_state_dict(resume_checkpoint["model_state"])
                    optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
                    _move_optimizer_state_to_device(optimizer, device)
                    scaler.load_state_dict(resume_checkpoint["scaler_state"])
                except Exception as exc:
                    raise RuntimeError(f"Failed to restore training state from {resume_from}: {exc}") from exc
                history = list(resume_checkpoint["history"])
                current_epoch = int(resume_checkpoint["current_epoch"])
                best_acc = float(resume_checkpoint["best_acc"])
                best_epoch = int(resume_checkpoint["best_epoch"])
                total_train_time_sec = float(resume_checkpoint["total_train_time_sec"])

            _print_header(cfg, device, paths, profile, resume_from)

            dump_csv(paths["metrics_path"], history, METRIC_FIELDS)

            epochs_configured = int(train_cfg["epochs"])
            warmup_epochs = int(_get(cfg, "train", "warmup_epochs", default=5))
            min_lr = float(_get(cfg, "train", "min_lr", default=1e-6))
            early_stop_patience = int(_get(cfg, "train", "early_stop_patience", default=10))
            min_delta = float(_get(cfg, "train", "min_delta", default=0.0))

            stale_epochs = 0
            if current_epoch > 0:
                stale_epochs = max(0, current_epoch - best_epoch)

            for epoch_idx in range(current_epoch, epochs_configured):
                epoch_number = epoch_idx + 1
                lr_now = _epoch_lr(base_lr, epoch_idx, epochs_configured, warmup_epochs, min_lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_now

                start = time.perf_counter()
                train_loss = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    criterion,
                    device,
                    mixup_fn=mixup_fn,
                    scaler=scaler,
                    max_grad_norm=max_grad_norm,
                )
                eval_metrics = evaluate(model, val_loader, device)
                epoch_time = time.perf_counter() - start
                total_train_time_sec += epoch_time

                val_acc = float(eval_metrics["top1"])
                val_top5 = float(eval_metrics["top5"])
                improved = val_acc > best_acc + min_delta
                if improved:
                    best_acc = val_acc
                    best_epoch = epoch_number
                    stale_epochs = 0
                    best_payload = {
                        **_build_common_payload(
                            cfg,
                            paths,
                            profile,
                            seed,
                            deterministic,
                            resume_from,
                            eval_split="val",
                        ),
                        "type": "best",
                        "best_acc": best_acc,
                        "best_epoch": best_epoch,
                        "current_epoch": epoch_number,
                        "model_state": _cpu_state_dict(model),
                        "evaluated_split": "val",
                    }
                    _save_checkpoint(paths["best_checkpoint_path"], best_payload)
                else:
                    stale_epochs += 1

                history.append(
                    {
                        "epoch": epoch_number,
                        "lr": lr_now,
                        "train_loss": float(train_loss),
                        "val_acc": val_acc,
                        "val_top5": val_top5,
                        "best_acc": best_acc,
                        "epoch_time_sec": epoch_time,
                    }
                )
                dump_csv(paths["metrics_path"], history, METRIC_FIELDS)

                last_payload = {
                    **_build_common_payload(
                        cfg,
                        paths,
                        profile,
                        seed,
                        deterministic,
                        resume_from,
                        eval_split="val",
                    ),
                    "type": "last",
                    "model_state": _cpu_state_dict(model),
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "current_epoch": epoch_number,
                    "best_acc": best_acc,
                    "best_epoch": best_epoch,
                    "history": history,
                    "total_train_time_sec": total_train_time_sec,
                    "last_epoch_metrics": history[-1],
                }
                _save_checkpoint(paths["last_checkpoint_path"], last_payload)

                print(
                    f"{epoch_number:5d} | {lr_now:10.6f} | {float(train_loss):10.4f} | "
                    f"{val_acc:10.2f} | {best_acc:10.2f} | {epoch_time:8.1f}s"
                )

                if early_stop_patience > 0 and stale_epochs >= early_stop_patience:
                    print(
                        f"early stop triggered at epoch {epoch_number} "
                        f"after {stale_epochs} stale epochs."
                    )
                    current_epoch = epoch_number
                    break

                current_epoch = epoch_number

            if current_epoch == 0 and history:
                current_epoch = int(history[-1]["epoch"])

            best_ckpt = _load_checkpoint(paths["best_checkpoint_path"])
            model.load_state_dict(best_ckpt["model_state"])
            eval_metrics = evaluate(model, val_loader, device)
            eval_result = {
                "model_name": cfg["model"]["name"],
                "checkpoint_path": str(paths["best_checkpoint_path"]),
                "dataset_root": str(Path(cfg["data"]["root"]).resolve()),
                "split": "val",
                "top1": float(eval_metrics["top1"]),
                "top5": float(eval_metrics["top5"]),
                "num_samples": int(eval_metrics["num_samples"]),
                "seed": seed,
                "deterministic": deterministic,
                "batch_size": int(cfg["data"]["batch_size"]),
                "img_size": int(cfg["data"]["img_size"]),
                "device": str(device),
            }
            dump_json(paths["eval_path"], eval_result)

            summary = {
                "model_name": cfg["model"]["name"],
                "dataset": cfg["data"]["dataset"],
                "dataset_root": str(Path(cfg["data"]["root"]).resolve()),
                "img_size": int(cfg["data"]["img_size"]),
                "batch_size": int(cfg["data"]["batch_size"]),
                "epochs_configured": int(cfg["train"]["epochs"]),
                "epochs_completed": int(current_epoch),
                "best_epoch": int(best_epoch),
                "best_val_acc": float(best_acc),
                "eval_top1": float(eval_result["top1"]),
                "eval_top5": float(eval_result["top5"]),
                "checkpoint_path": str(paths["best_checkpoint_path"]),
                "best_checkpoint_path": str(paths["best_checkpoint_path"]),
                "last_checkpoint_path": str(paths["last_checkpoint_path"]),
                "log_path": str(paths["log_path"]),
                "metrics_path": str(paths["metrics_path"]),
                "summary_path": str(paths["summary_path"]),
                "eval_path": str(paths["eval_path"]),
                "seed": seed,
                "deterministic": deterministic,
                "pretrained": bool(cfg["model"]["pretrained"]),
                "mixup": float(cfg["train"]["mixup_alpha"]),
                "cutmix": float(cfg["train"]["cutmix_alpha"]),
                "mixup_alpha": float(cfg["train"]["mixup_alpha"]),
                "cutmix_alpha": float(cfg["train"]["cutmix_alpha"]),
                "label_smoothing": float(cfg["train"]["label_smoothing"]),
                "weight_decay": float(cfg["train"]["weight_decay"]),
                "drop_path_rate": float(cfg["model"]["drop_path_rate"]),
                "total_train_time_sec": float(total_train_time_sec),
                "Params (M)": f"{profile['params_m']:.2f}",
                "FLOPs (G)": f"{profile['flops_g']:.2f}" if profile["flops_g"] is not None else "N/A",
                "FLOPs note": profile["flops_note"],
                "resume_from": str(resume_from) if resume_from is not None else None,
                "config_path": cfg.get("_config_path"),
                "run_id": paths["run_id"],
                "date_str": paths["date_str"],
                "eval_command": (
                    f"python -u scripts/test.py --config {args.config} "
                    f"--checkpoint {paths['best_checkpoint_path']} --split val"
                ),
                "train_command": f"python -u scripts/train.py --config {args.config}"
                + (f" --resume {args.resume}" if args.resume else ""),
            }
            summary_md = _render_summary_md(summary)
            paths["summary_path"].write_text(summary_md, encoding="utf-8")

            # Persist the summary fields back into the last checkpoint for easier resume/debugging.
            last_ckpt = _load_checkpoint(paths["last_checkpoint_path"])
            last_ckpt["summary_path"] = str(paths["summary_path"])
            last_ckpt["eval_result_path"] = str(paths["eval_path"])
            last_ckpt["best_checkpoint_path"] = str(paths["best_checkpoint_path"])
            last_ckpt["last_checkpoint_path"] = str(paths["last_checkpoint_path"])
            last_ckpt["epochs_completed"] = int(current_epoch)
            last_ckpt["best_val_acc"] = float(best_acc)
            last_ckpt["best_epoch"] = int(best_epoch)
            last_ckpt["total_train_time_sec"] = float(total_train_time_sec)
            _save_checkpoint(paths["last_checkpoint_path"], last_ckpt)

            print("-" * 80)
            print(f"finished. best val acc: {best_acc:.2f}% at epoch {best_epoch}")
            print(f"epochs completed: {current_epoch}/{epochs_configured}")
            print(f"best checkpoint saved to: {paths['best_checkpoint_path']}")
            print(f"last checkpoint saved to: {paths['last_checkpoint_path']}")
            print(f"metrics saved to: {paths['metrics_path']}")
            print(f"summary saved to: {paths['summary_path']}")
            print(f"eval result saved to: {paths['eval_path']}")
            print(f"log saved to: {paths['log_path']}")


if __name__ == "__main__":
    main()
