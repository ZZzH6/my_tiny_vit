from __future__ import annotations

import io
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


def load_words_map(dataset_root: Path) -> dict[str, str]:
    words_path = dataset_root / "words.txt"
    mapping: dict[str, str] = {}
    if not words_path.exists():
        return mapping
    with words_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            wnid, label = line.split("\t", 1)
            mapping[wnid] = label.split(",")[0].strip()
    return mapping


def get_class_names(dataset_root: Path) -> list[str]:
    train_root = dataset_root / "train"
    if not train_root.exists():
        return []
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


@st.cache_data(show_spinner=False)
def get_model_meta(model_key: str) -> dict[str, Any]:
    spec = MODEL_SPECS[model_key]
    cfg = read_yaml(spec.config_path)
    summary = parse_summary_markdown(spec.summary_path)
    dataset_root = Path(summary.get("dataset_root", ROOT / "dataset/tiny-imagenet-200"))
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
    checkpoint = torch.load(spec.checkpoint_path, map_location="cpu")
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
        wnid = class_names[index] if index < len(class_names) else str(index)
        top_items.append(
            {
                "index": int(index),
                "wnid": wnid,
                "label": format_label(wnid, label_display),
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
