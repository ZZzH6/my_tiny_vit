from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from models import build_model_from_cfg


def _get(cfg, *keys, default=None):
    cur = cfg
    for key in keys:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def build_teacher_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    path = Path(checkpoint_path).expanduser().resolve()
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Teacher checkpoint not found: {path}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to load teacher checkpoint from {path}: {exc}") from exc

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Teacher checkpoint at {path} must be a dict, got {type(checkpoint).__name__}")

    teacher_cfg = checkpoint.get("config")
    if not isinstance(teacher_cfg, dict):
        raise ValueError(f"Teacher checkpoint at {path} is missing config")

    teacher_model_cfg = teacher_cfg.get("model")
    if not isinstance(teacher_model_cfg, dict):
        raise ValueError(f"Teacher checkpoint at {path} is missing config.model")

    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, dict):
        raise ValueError(f"Teacher checkpoint at {path} is missing model_state")

    teacher_model = build_model_from_cfg(teacher_model_cfg, pretrained_override=False)
    teacher_model.load_state_dict(model_state)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    for parameter in teacher_model.parameters():
        parameter.requires_grad_(False)

    metadata = {
        "checkpoint_path": str(path),
        "model_name": str(_get(teacher_cfg, "model", "name", default="unknown")),
        "num_classes": int(_get(teacher_cfg, "model", "num_classes", default=-1)),
        "img_size": _get(teacher_cfg, "data", "img_size", default=None),
        "best_acc": checkpoint.get("best_acc"),
        "best_epoch": checkpoint.get("best_epoch"),
    }
    return teacher_model, metadata


def compute_soft_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError(f"distillation temperature must be positive, got {temperature}")

    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature * temperature)


def compute_hard_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> torch.Tensor:
    return F.cross_entropy(student_logits, teacher_logits.argmax(dim=1))
