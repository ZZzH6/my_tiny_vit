from __future__ import annotations

import io
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import torch
import yaml
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.build_loader import IMAGENET_MEAN, IMAGENET_STD
from models import build_model_from_cfg

MIN_CHECKPOINT_BYTES = 1024
DEFAULT_DATASET_ROOT = ROOT / "dataset/tiny-imagenet-200"
CLASS_INDEX_PATH = Path(__file__).with_name("class_index_imagenet_tiny.json")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    title: str
    badge: str
    config_path: Path
    summary_path: Path
    checkpoint_path: Path
    role: str
    note: str
    color: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "baseline_224": ModelSpec(
        key="baseline_224",
        title="DeiT 224 Baseline",
        badge="Anchor",
        config_path=ROOT / "configs/deit_tiny_baseline.yaml",
        summary_path=ROOT / "results/summary/20260422/deit_tiny_baseline_20260422_233858.md",
        checkpoint_path=ROOT / "results/checkpoints/20260422/deit_tiny_baseline_20260422_233858_best.pt",
        role="标准锚点模型",
        note="timm deit_tiny_patch16_224，方案A 224 输入基线。",
        color="#4da3ff",
    ),
    "baseline_112": ModelSpec(
        key="baseline_112",
        title="Patch8 112 Baseline",
        badge="Mainline",
        config_path=ROOT / "configs/deit_tiny_patch8_112_baseline.yaml",
        summary_path=ROOT / "results/summary/20260421/deit_tiny_patch8_112_baseline_20260421_003247.md",
        checkpoint_path=ROOT / "results/checkpoints/20260421/deit_tiny_patch8_112_baseline_20260421_003247_best.pt",
        role="112 主线 baseline",
        note="64 -> 112，patch8，作为轻量化主线对照。",
        color="#ff9f40",
    ),
    "teacher_final": ModelSpec(
        key="teacher_final",
        title="Teacher Two-Stage",
        badge="Teacher",
        config_path=ROOT / "configs/deit_tiny_patch8_112_overlap_patch12_teacher_twostage.yaml",
        summary_path=ROOT / "results/summary/20260423/deit_tiny_patch8_112_overlap_patch12_teacher_twostage_20260423_232954.md",
        checkpoint_path=ROOT / "results/checkpoints/20260423/deit_tiny_patch8_112_overlap_patch12_teacher_twostage_20260423_232954_best.pt",
        role="最终 teacher",
        note="overlap patch12 + strong recipe + low-reg refine。",
        color="#f97316",
    ),
    "student_final": ModelSpec(
        key="student_final",
        title="Student D10 Final",
        badge="Deploy",
        config_path=ROOT / "configs/deit_tiny_patch8_112_student_depth10_deit_harddistill_twostage.yaml",
        summary_path=ROOT / "results/summary/20260424/deit_tiny_patch8_112_student_depth10_deit_harddistill_twostage_20260424_180111.md",
        checkpoint_path=ROOT / "results/checkpoints/20260424/deit_tiny_patch8_112_student_depth10_deit_harddistill_twostage_20260424_180111_best.pt",
        role="最终学生模型",
        note="depth10 + hard distill + two-stage refine，当前部署主模型。",
        color="#22c55e",
    ),
}

SUMMARY_STATIC_FALLBACKS: dict[str, dict[str, float | int]] = {
    "baseline_224": {
        "img_size": 224,
        "best_val_acc": 77.37,
        "eval_top5": 92.59,
        "params_m": 5.56,
        "flops_g": 2.149395456,
    },
    "baseline_112": {
        "img_size": 112,
        "best_val_acc": 79.46,
        "eval_top5": 93.81,
        "params_m": 5.45,
        "flops_g": 2.106043392,
    },
    "teacher_final": {
        "img_size": 112,
        "best_val_acc": 80.18,
        "eval_top5": 94.07,
        "params_m": 5.50,
        "flops_g": 2.124106752,
    },
    "student_final": {
        "img_size": 112,
        "best_val_acc": 79.41,
        "eval_top5": 93.28,
        "params_m": 4.60,
        "flops_g": 1.766381568,
    },
}


class CheckpointLoadError(RuntimeError):
    """Raised when a deployment checkpoint cannot be safely loaded."""


def parse_summary_markdown(path: Path) -> dict[str, str]:
    summary: dict[str, str] = {}
    if not path.exists():
        return summary
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("- ") or ":" not in line:
            continue
        key, value = line[2:].split(":", 1)
        summary[key.strip()] = value.strip()
    return summary


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def merge_summary_fallbacks(model_key: str, summary: dict[str, str]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    merged.update(SUMMARY_STATIC_FALLBACKS.get(model_key, {}))
    merged.update(summary)
    return merged


def resolve_dataset_root(summary: dict[str, str]) -> Path:
    raw_root = str(summary.get("dataset_root", "")).strip()
    candidates: list[Path] = []
    if raw_root:
        summary_root = Path(raw_root)
        candidates.append(summary_root)
        if not summary_root.is_absolute():
            candidates.append(ROOT / summary_root)
    candidates.append(DEFAULT_DATASET_ROOT)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return DEFAULT_DATASET_ROOT


def load_class_index_names() -> list[str]:
    if not CLASS_INDEX_PATH.exists():
        return []
    try:
        payload = json.loads(CLASS_INDEX_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict):
        return []
    class_names = payload.get("class_names", [])
    if not isinstance(class_names, list):
        return []
    cleaned = [str(name).strip() for name in class_names if str(name).strip()]
    return cleaned


def load_wnids(dataset_root: Path) -> list[str]:
    wnids_path = dataset_root / "wnids.txt"
    if not wnids_path.exists():
        return []
    return [line.strip() for line in wnids_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_words_map(dataset_root: Path) -> dict[str, str]:
    words_path = dataset_root / "words.txt"
    mapping: dict[str, str] = {}
    if not words_path.exists():
        return mapping
    with words_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                wnid, label_text = line.split("\t", 1)
            else:
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                wnid, label_text = parts
            wnid = wnid.strip()
            label_text = label_text.strip()
            if not wnid or not label_text:
                continue
            primary_label = label_text.split(",")[0].strip() or label_text
            mapping.setdefault(wnid, primary_label)
    return mapping


def get_class_names(dataset_root: Path) -> list[str]:
    class_index_names = load_class_index_names()
    if class_index_names:
        return class_index_names
    train_root = dataset_root / "train"
    if not train_root.exists():
        return load_wnids(dataset_root)
    return sorted(path.name for path in train_root.iterdir() if path.is_dir())


def build_label_display(dataset_root: Path) -> dict[str, str]:
    names = get_class_names(dataset_root)
    words_map = load_words_map(dataset_root)
    return {name: words_map.get(name, name) for name in names}


def resolve_interpolation(name: str) -> InterpolationMode:
    mapping = {
        "nearest": InterpolationMode.NEAREST,
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "box": InterpolationMode.BOX,
        "hamming": InterpolationMode.HAMMING,
        "lanczos": InterpolationMode.LANCZOS,
    }
    return mapping.get(str(name).strip().lower(), InterpolationMode.BICUBIC)


def build_eval_transform(cfg: dict[str, Any]):
    data_cfg = cfg["data"]
    img_size = int(data_cfg.get("img_size", 224))
    use_imagenet_eval = bool(data_cfg.get("use_imagenet_eval", False))
    eval_crop_ratio = float(data_cfg.get("eval_crop_ratio", 0.875))
    interpolation = resolve_interpolation(str(data_cfg.get("eval_interpolation", "bicubic")))
    if use_imagenet_eval:
        resize_size = int(img_size / eval_crop_ratio)
        return transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=interpolation, antialias=True),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size), interpolation=interpolation, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def format_file_size(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "N/A"
    size = float(size_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size_bytes} B"


def build_checkpoint_error_message(path: Path, size_bytes: int | None, reason: str) -> str:
    size_text = format_file_size(size_bytes)
    return "\n".join(
        [
            "Checkpoint load failed.",
            f"checkpoint_path: {path}",
            f"file_size: {size_text} ({size_bytes if size_bytes is not None else 'N/A'} bytes)",
            f"torch_version: {torch.__version__}",
            f"reason: {reason}",
        ]
    )


def validate_checkpoint_file(path: Path) -> int:
    if not path.exists():
        raise CheckpointLoadError(build_checkpoint_error_message(path, None, "checkpoint file does not exist"))

    size_bytes = path.stat().st_size
    if size_bytes <= 0:
        raise CheckpointLoadError(build_checkpoint_error_message(path, size_bytes, "checkpoint file is empty"))

    if size_bytes < MIN_CHECKPOINT_BYTES:
        prefix = path.read_bytes()[:256]
        if b"git-lfs.github.com/spec/v1" in prefix:
            raise CheckpointLoadError(
                build_checkpoint_error_message(
                    path,
                    size_bytes,
                    "checkpoint file is a Git LFS pointer, not the real binary payload",
                )
            )
        raise CheckpointLoadError(
            build_checkpoint_error_message(
                path,
                size_bytes,
                f"checkpoint file is unexpectedly small (< {MIN_CHECKPOINT_BYTES} bytes)",
            )
        )

    return size_bytes


def load_checkpoint_file(path: Path) -> dict[str, Any]:
    size_bytes = validate_checkpoint_file(path)
    try:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError as exc:
            if "weights_only" not in str(exc):
                raise
            checkpoint = torch.load(path, map_location="cpu")
    except CheckpointLoadError:
        raise
    except Exception as exc:
        raise CheckpointLoadError(
            build_checkpoint_error_message(
                path,
                size_bytes,
                f"torch.load raised {type(exc).__name__}: {exc}",
            )
        ) from exc

    if not isinstance(checkpoint, dict):
        raise CheckpointLoadError(
            build_checkpoint_error_message(
                path,
                size_bytes,
                f"unexpected checkpoint type: {type(checkpoint).__name__}",
            )
        )
    if "model_state" not in checkpoint:
        raise CheckpointLoadError(
            build_checkpoint_error_message(path, size_bytes, "checkpoint missing required key: model_state")
        )
    return checkpoint


@st.cache_data(show_spinner=False)
def get_model_meta(model_key: str) -> dict[str, Any]:
    spec = MODEL_SPECS[model_key]
    cfg = read_yaml(spec.config_path)
    summary = merge_summary_fallbacks(model_key, parse_summary_markdown(spec.summary_path))
    dataset_root = resolve_dataset_root(summary)
    meta = {
        "spec": spec,
        "cfg": cfg,
        "summary": summary,
        "dataset_root": dataset_root,
        "class_names": get_class_names(dataset_root),
        "label_display": build_label_display(dataset_root),
    }
    return meta


@st.cache_resource(show_spinner=False)
def load_model(model_key: str, device: str):
    meta = get_model_meta(model_key)
    cfg = meta["cfg"]
    spec: ModelSpec = meta["spec"]
    model = build_model_from_cfg(cfg["model"], pretrained_override=False)
    checkpoint = load_checkpoint_file(spec.checkpoint_path)
    model.load_state_dict(checkpoint["model_state"])
    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()
    return model


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def format_label(wnid: str, label_display: dict[str, str]) -> str:
    human = label_display.get(wnid, wnid)
    return f"{human} ({wnid})" if human != wnid else wnid


def resolve_class_name(index: int, class_names: list[str]) -> str | None:
    if 0 <= index < len(class_names):
        return class_names[index]
    return None


def format_prediction_label(index: int, class_names: list[str], label_display: dict[str, str]) -> tuple[str, str]:
    wnid = resolve_class_name(index, class_names)
    if wnid is None:
        return str(index), str(index)
    return wnid, format_label(wnid, label_display)


def predict_image(
    model_key: str,
    image: Image.Image,
    device: str,
    topk: int = 5,
) -> dict[str, Any]:
    meta = get_model_meta(model_key)
    cfg = meta["cfg"]
    class_names: list[str] = meta["class_names"]
    label_display: dict[str, str] = meta["label_display"]
    model = load_model(model_key, device)
    transform = build_eval_transform(cfg)

    rgb_image = image.convert("RGB")
    tensor = transform(rgb_image).unsqueeze(0)
    torch_device = torch.device(device)
    tensor = tensor.to(torch_device)

    with torch.no_grad():
        _sync_if_cuda(torch_device)
        start = time.perf_counter()
        logits = model(tensor)
        _sync_if_cuda(torch_device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        probs = torch.softmax(logits, dim=1)
        scores, indices = probs.topk(min(topk, probs.shape[1]), dim=1)

    top_items = []
    for index, score in zip(indices[0].tolist(), scores[0].tolist()):
        wnid, label = format_prediction_label(int(index), class_names, label_display)
        top_items.append(
            {
                "index": int(index),
                "wnid": wnid,
                "label": label,
                "prob": float(score),
            }
        )

    summary = meta["summary"]
    return {
        "model_key": model_key,
        "title": meta["spec"].title,
        "badge": meta["spec"].badge,
        "device": device,
        "elapsed_ms": elapsed_ms,
        "top_items": top_items,
        "best_val_acc": float(summary.get("best_val_acc", 0.0)),
        "params_m": float(summary.get("params_m", 0.0)),
        "flops_g": float(summary.get("flops_g", 0.0)),
        "img_size": int(summary.get("img_size", cfg["data"]["img_size"])),
        "eval_top5": float(summary.get("eval_top5", 0.0)),
        "note": meta["spec"].note,
    }


def predict_batch(
    model_key: str,
    files: list[Any],
    device: str,
    topk: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for file in files:
        image = Image.open(io.BytesIO(file.getvalue())).convert("RGB")
        result = predict_image(model_key, image=image, device=device, topk=topk)
        top1 = result["top_items"][0]
        rows.append(
            {
                "image": file.name,
                "model": result["title"],
                "top1_label": top1["label"],
                "top1_prob": round(top1["prob"], 4),
                "latency_ms": round(result["elapsed_ms"], 2),
                "top5": " | ".join(
                    f"{item['label']} {item['prob']:.3f}" for item in result["top_items"]
                ),
            }
        )
    return pd.DataFrame(rows)


def build_model_catalog() -> pd.DataFrame:
    rows = []
    for key in MODEL_SPECS:
        meta = get_model_meta(key)
        summary = meta["summary"]
        spec: ModelSpec = meta["spec"]
        rows.append(
            {
                "Model": spec.title,
                "Role": spec.role,
                "Input": int(summary.get("img_size", 0)),
                "Top-1 (%)": float(summary.get("best_val_acc", 0.0)),
                "Params (M)": float(summary.get("params_m", 0.0)),
                "FLOPs (G)": float(summary.get("flops_g", 0.0)),
                "Note": spec.note,
            }
        )
    return pd.DataFrame(rows)


def build_scatter_frame() -> pd.DataFrame:
    rows = []
    for key in MODEL_SPECS:
        meta = get_model_meta(key)
        summary = meta["summary"]
        rows.append(
            {
                "name": meta["spec"].title,
                "top1": float(summary.get("best_val_acc", 0.0)),
                "flops": float(summary.get("flops_g", 0.0)),
                "params": float(summary.get("params_m", 0.0)),
                "color": meta["spec"].color,
            }
        )
    return pd.DataFrame(rows)


def latest_thesis_visual_dir() -> Path | None:
    figure_root = ROOT / "results" / "figures"
    candidates = sorted(figure_root.glob("*/thesis_visuals"))
    return candidates[-1] if candidates else None


def load_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def format_metric(value: float, digits: int = 2, suffix: str = "") -> str:
    return f"{value:.{digits}f}{suffix}"
