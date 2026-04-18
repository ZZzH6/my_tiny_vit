from __future__ import annotations

import argparse
import contextlib
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from timm.loss import SoftTargetCrossEntropy
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.build_loader import build_loader
from engine.distillation import build_teacher_from_checkpoint
from engine.evaluator import evaluate
from engine.trainer import train_one_epoch
from models import build_model_from_cfg
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
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Initialize a new run from a checkpoint without reusing the original artifacts.",
    )
    return parser.parse_args()


class _TeeWriter:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
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


def _normalize_train_stages(train_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    raw_stages = train_cfg.get("stages")
    if raw_stages is None:
        return [
            {
                "name": "main",
                "epochs": int(train_cfg["epochs"]),
                "lr": float(train_cfg["lr"]),
                "warmup_epochs": int(train_cfg.get("warmup_epochs", 5)),
                "min_lr": float(train_cfg.get("min_lr", 1.0e-6)),
            }
        ]

    if not isinstance(raw_stages, list) or len(raw_stages) == 0:
        raise ValueError("train.stages must be a non-empty list when provided.")

    stages: list[dict[str, Any]] = []
    total_epochs = 0
    default_lr = float(train_cfg["lr"])
    default_warmup_epochs = int(train_cfg.get("warmup_epochs", 5))
    default_min_lr = float(train_cfg.get("min_lr", 1.0e-6))
    for stage_idx, raw_stage in enumerate(raw_stages, start=1):
        if not isinstance(raw_stage, dict):
            raise ValueError(f"train.stages[{stage_idx - 1}] must be a dict.")

        stage_epochs = int(raw_stage["epochs"])
        if stage_epochs <= 0:
            raise ValueError(f"train.stages[{stage_idx - 1}].epochs must be positive.")

        stage = {
            "name": str(raw_stage.get("name", f"stage{stage_idx}")),
            "epochs": stage_epochs,
            "lr": float(raw_stage.get("lr", default_lr)),
            "warmup_epochs": int(raw_stage.get("warmup_epochs", default_warmup_epochs)),
            "min_lr": float(raw_stage.get("min_lr", default_min_lr)),
        }
        stages.append(stage)
        total_epochs += stage_epochs

    configured_epochs = train_cfg.get("epochs")
    if configured_epochs is not None and int(configured_epochs) != total_epochs:
        raise ValueError(
            "train.epochs must equal the sum of train.stages[*].epochs when train.stages is set."
        )
    return stages


def _stage_for_epoch(train_stages: list[dict[str, Any]], epoch_idx: int) -> tuple[int, dict[str, Any], int]:
    stage_start_epoch = 0
    for stage_idx, stage in enumerate(train_stages, start=1):
        stage_end_epoch = stage_start_epoch + int(stage["epochs"])
        if epoch_idx < stage_end_epoch:
            local_epoch_idx = epoch_idx - stage_start_epoch
            return stage_idx, stage, local_epoch_idx
        stage_start_epoch = stage_end_epoch
    raise ValueError(f"Epoch index {epoch_idx} is out of range for the configured training stages.")


def _build_criterion(cfg, use_mixup: bool):
    if use_mixup:
        return SoftTargetCrossEntropy()
    return torch.nn.CrossEntropyLoss(
        label_smoothing=float(_get(cfg, "train", "label_smoothing", default=0.0))
    )


def _resolve_path(value: str | None, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _get_distillation_cfg(cfg, base_dir: Path) -> dict[str, Any] | None:
    raw_cfg = _get(cfg, "train", "distillation", default=None)
    if raw_cfg is None:
        return None
    if not isinstance(raw_cfg, dict):
        raise ValueError("train.distillation must be a dict when provided.")
    if not bool(raw_cfg.get("enabled", True)):
        return None

    teacher_checkpoint_raw = raw_cfg.get("teacher_checkpoint")
    if not teacher_checkpoint_raw:
        raise ValueError("train.distillation.teacher_checkpoint is required when distillation is enabled.")

    method = str(raw_cfg.get("method", "logit"))
    distillation_type = str(raw_cfg.get("type", "soft"))
    alpha = float(raw_cfg.get("alpha", 0.5))
    temperature = float(raw_cfg.get("temperature", 4.0))
    if method not in {"logit", "deit"}:
        raise ValueError(f"train.distillation.method must be one of ['logit', 'deit'], got {method}")
    if distillation_type not in {"soft", "hard"}:
        raise ValueError(
            f"train.distillation.type must be one of ['soft', 'hard'], got {distillation_type}"
        )
    if method == "logit" and distillation_type != "soft":
        raise ValueError("train.distillation.type must be 'soft' when train.distillation.method='logit'.")
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"train.distillation.alpha must be in [0, 1], got {alpha}")
    if distillation_type == "soft" and temperature <= 0.0:
        raise ValueError(f"train.distillation.temperature must be positive, got {temperature}")

    return {
        "teacher_checkpoint": str(_resolve_path(str(teacher_checkpoint_raw), base_dir)),
        "method": method,
        "type": distillation_type,
        "alpha": alpha,
        "temperature": temperature,
    }


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
        "# DeiT-Tiny Experiment Summary",
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
        f"- init_checkpoint: {summary['init_checkpoint']}",
        f"- distillation_enabled: {summary['distillation_enabled']}",
        f"- distillation_method: {summary['distillation_method']}",
        f"- distillation_type: {summary['distillation_type']}",
        f"- teacher_checkpoint: {summary['teacher_checkpoint']}",
        f"- distillation_alpha: {summary['distillation_alpha']}",
        f"- distillation_temperature: {summary['distillation_temperature']}",
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

    resume_value = args.resume or _get(cfg, "train", "resume_from", default=None)
    init_checkpoint_value = args.init_checkpoint or _get(cfg, "train", "init_checkpoint", default=None)
    if resume_value and init_checkpoint_value:
        raise ValueError("Use either resume or init_checkpoint, not both.")

    resume_path = _resolve_path(resume_value, ROOT)
    init_checkpoint_path = _resolve_path(init_checkpoint_value, ROOT)
    distillation_cfg = _get_distillation_cfg(cfg, ROOT)
    paths, resume_checkpoint = _resolve_paths(config_path.stem, resume_path)
    init_checkpoint = _load_checkpoint(init_checkpoint_path) if init_checkpoint_path is not None else None
    for key in ARTIFACT_KEYS:
        Path(paths[key]).parent.mkdir(parents=True, exist_ok=True)

    with open(Path(paths["log_path"]), "a", encoding="utf-8") as log_file:
        tee_out = _TeeWriter(sys.__stdout__, log_file)
        tee_err = _TeeWriter(sys.__stderr__, log_file)
        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            model_cfg = cfg["model"]
            data_cfg = cfg["data"]
            train_cfg = cfg["train"]
            train_stages = _normalize_train_stages(train_cfg)
            epochs = sum(int(stage["epochs"]) for stage in train_stages)

            model_init_pretrained = (
                bool(model_cfg["pretrained"]) and resume_checkpoint is None and init_checkpoint is None
            )
            model = build_model_from_cfg(model_cfg, pretrained_override=model_init_pretrained)
            profile = profile_model(
                model,
                input_size=(3, int(data_cfg["img_size"]), int(data_cfg["img_size"])),
            )
            model = model.to(device)
            teacher_model = None
            teacher_info = None
            if distillation_cfg is not None:
                teacher_model, teacher_info = build_teacher_from_checkpoint(
                    distillation_cfg["teacher_checkpoint"],
                    device,
                )
                teacher_img_size = teacher_info.get("img_size")
                if teacher_img_size is not None and int(teacher_img_size) != int(data_cfg["img_size"]):
                    raise ValueError(
                        "Teacher img_size does not match student img_size: "
                        f"{teacher_img_size} vs {data_cfg['img_size']}"
                    )
                teacher_num_classes = int(teacher_info.get("num_classes", -1))
                if teacher_num_classes != int(model_cfg["num_classes"]):
                    raise ValueError(
                        "Teacher num_classes does not match student num_classes: "
                        f"{teacher_num_classes} vs {model_cfg['num_classes']}"
                    )
                if distillation_cfg["method"] == "deit":
                    if not hasattr(model, "set_distilled_training"):
                        raise ValueError(
                            "DeiT-style distillation requires a distilled student model. "
                            "Set model.distilled: true in the config."
                        )
                    model.set_distilled_training(True)

            train_loader, val_loader, mixup_fn = build_loader(cfg)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(train_stages[0]["lr"]),
                weight_decay=float(train_cfg["weight_decay"]),
            )
            criterion = _build_criterion(cfg, use_mixup=mixup_fn is not None)
            amp_enabled = bool(_get(cfg, "train", "amp", default=True)) and device.type == "cuda"
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
            max_grad_norm = float(_get(cfg, "train", "max_grad_norm", default=1.0))

            history: list[dict[str, Any]] = []
            start_epoch = 0
            best_acc = float("-inf")
            best_epoch = 0
            total_train_time_sec = 0.0

            if init_checkpoint is not None:
                model.load_state_dict(init_checkpoint["model_state"])

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
            print("Tiny-ImageNet | DeiT-Tiny training")
            print("=" * 80)
            print(f"config     : {cfg['_config_path']}")
            print(f"device     : {device}")
            print(f"dataset    : {data_cfg['dataset']}")
            print(f"data root  : {data_cfg['root']}")
            print(f"img size   : {data_cfg['img_size']}")
            print(f"batch size : {data_cfg['batch_size']}")
            print(f"epochs     : {epochs}")
            print(f"lr         : {train_stages[0]['lr']}")
            print(f"wd         : {train_cfg['weight_decay']}")
            print(f"pretrained : {model_init_pretrained}")
            print(f"init ckpt  : {init_checkpoint_path if init_checkpoint_path is not None else 'N/A'}")
            print(f"resume     : {resume_path if resume_path is not None else 'N/A'}")
            print(f"Params     : {profile['params_m']:.2f}M")
            if profile["flops_g"] is not None:
                print(f"FLOPs      : {profile['flops_g']:.2f}G")
            else:
                print(f"FLOPs      : N/A ({profile['flops_note']})")
            if hasattr(model, "distilled_training"):
                print(f"student    : {'distilled' if bool(_get(cfg, 'model', 'distilled', default=False)) else 'standard'}")
                init_source = getattr(model, "pretrained_init_source", None)
                if init_source is not None:
                    print(f"init source: {init_source}")
            if teacher_info is not None:
                print("distill    : enabled")
                print(f"teacher    : {teacher_info['checkpoint_path']}")
                print(
                    f"kd method  : {distillation_cfg['method']} / {distillation_cfg['type']}"
                )
                print(
                    f"kd alpha/T : {distillation_cfg['alpha']:.2f} / "
                    f"{distillation_cfg['temperature']:.2f}"
                )
                if teacher_info.get("best_acc") is not None and teacher_info.get("best_epoch") is not None:
                    print(
                        f"teacher top1: {float(teacher_info['best_acc']):.2f}% "
                        f"(epoch {int(teacher_info['best_epoch'])})"
                    )
            else:
                print("distill    : disabled")
            if len(train_stages) > 1:
                print("stages     :")
                for stage_idx, stage in enumerate(train_stages, start=1):
                    print(
                        "  "
                        f"{stage_idx}. {stage['name']} | epochs={stage['epochs']} | lr={stage['lr']} | "
                        f"warmup={stage['warmup_epochs']} | min_lr={stage['min_lr']}"
                    )
            print("-" * 80)
            print(
                f"{'epoch':>5} | {'lr':>10} | {'train_loss':>10} | "
                f"{'val_acc(%)':>10} | {'best_acc(%)':>10} | {'time':>8}"
            )
            print("-" * 80)

            previous_stage_idx = None
            for epoch_idx in range(start_epoch, epochs):
                epoch_number = epoch_idx + 1
                stage_idx, stage_cfg, local_epoch_idx = _stage_for_epoch(train_stages, epoch_idx)
                if stage_idx != previous_stage_idx:
                    print(
                        f"enter stage {stage_idx}/{len(train_stages)}: {stage_cfg['name']} "
                        f"(epochs={stage_cfg['epochs']}, lr={stage_cfg['lr']}, "
                        f"warmup={stage_cfg['warmup_epochs']}, min_lr={stage_cfg['min_lr']})"
                    )
                    previous_stage_idx = stage_idx

                lr_now = _epoch_lr(
                    stage_cfg["lr"],
                    local_epoch_idx,
                    stage_cfg["epochs"],
                    stage_cfg["warmup_epochs"],
                    stage_cfg["min_lr"],
                )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_now

                start_time = time.perf_counter()
                train_loss = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    criterion,
                    device,
                    mixup_fn=mixup_fn,
                    scaler=scaler,
                    max_grad_norm=max_grad_norm,
                    teacher_model=teacher_model,
                    distillation_alpha=0.0 if distillation_cfg is None else distillation_cfg["alpha"],
                    distillation_temperature=1.0
                    if distillation_cfg is None
                    else distillation_cfg["temperature"],
                    distillation_method="logit"
                    if distillation_cfg is None
                    else distillation_cfg["method"],
                    distillation_type="soft"
                    if distillation_cfg is None
                    else distillation_cfg["type"],
                )
                eval_metrics = evaluate(model, val_loader, device)
                epoch_time = time.perf_counter() - start_time
                total_train_time_sec += epoch_time

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
                        "train_loss": float(train_loss),
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
                    f"{epoch_number:5d} | {lr_now:10.6f} | {float(train_loss):10.4f} | "
                    f"{val_acc:10.2f} | {best_acc:10.2f} | {epoch_time:8.1f}s"
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
                "init_checkpoint": str(init_checkpoint_path) if init_checkpoint_path is not None else "N/A",
                "distillation_enabled": teacher_info is not None,
                "distillation_method": (
                    str(distillation_cfg["method"]) if distillation_cfg is not None else "N/A"
                ),
                "distillation_type": (
                    str(distillation_cfg["type"]) if distillation_cfg is not None else "N/A"
                ),
                "teacher_checkpoint": (
                    teacher_info["checkpoint_path"] if teacher_info is not None else "N/A"
                ),
                "distillation_alpha": (
                    float(distillation_cfg["alpha"]) if distillation_cfg is not None else 0.0
                ),
                "distillation_temperature": (
                    float(distillation_cfg["temperature"]) if distillation_cfg is not None else 1.0
                ),
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
                "train_command": (
                    f"python -u scripts/train.py --config {args.config}"
                    + (f" --init-checkpoint {init_checkpoint_value}" if init_checkpoint_value else "")
                    + (f" --resume {resume_value}" if resume_value else "")
                ),
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
