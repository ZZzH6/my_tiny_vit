from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.build_loader import build_eval_loader
from engine.evaluator import evaluate
from models.baseline_models import build_model
from utils.artifacts import build_run_paths, dump_json
from utils.model_zoo import resolve_model_zoo_best_checkpoint
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
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional path to a specific checkpoint. If omitted, use results/models/<model_name>/best.pt.",
    )
    parser.add_argument("--split", default="val", choices=["train", "val"])
    return parser.parse_args()


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


def _print_result(result: dict[str, Any], eval_path: Path) -> None:
    print("=" * 80)
    print("Tiny-ImageNet-200 | DeiT-Tiny evaluation")
    print("=" * 80)
    print(f"model_name    : {result['model_name']}")
    print(f"checkpoint    : {result['checkpoint_path']}")
    print(f"checkpoint src: {result['checkpoint_source']}")
    print(f"dataset_root  : {result['dataset_root']}")
    print(f"split         : {result['split']}")
    print(f"top1          : {result['top1']:.2f}%")
    print(f"top5          : {result['top5']:.2f}%")
    print(f"num_samples   : {result['num_samples']}")
    print(f"eval file     : {eval_path}")
    print("=" * 80)


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
    if device_cfg == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    else:
        checkpoint_path, _ = resolve_model_zoo_best_checkpoint(ROOT / "results", model_cfg["name"])

    checkpoint = _load_checkpoint(checkpoint_path)
    _require_keys(checkpoint, ["model_state"], f"checkpoint {checkpoint_path}")

    model = build_model(
        model_name=model_cfg["name"],
        num_classes=int(model_cfg["num_classes"]),
        pretrained=bool(model_cfg["pretrained"]),
        drop_path_rate=float(_get(cfg, "model", "drop_path_rate", default=0.1)),
        drop_rate=float(_get(cfg, "model", "drop_rate", default=0.0)),
        attn_drop_rate=float(_get(cfg, "model", "attn_drop_rate", default=0.0)),
    ).to(device)

    try:
        model.load_state_dict(checkpoint["model_state"])
    except Exception as exc:
        raise RuntimeError(f"Failed to load model weights from {checkpoint_path}: {exc}") from exc

    eval_loader, eval_dataset = build_eval_loader(cfg, split=args.split)
    eval_metrics = evaluate(model, eval_loader, device)

    run_paths = build_run_paths(ROOT / "results", Path(args.config).stem, eval_split=args.split)
    run_paths["eval_path"].parent.mkdir(parents=True, exist_ok=True)

    result = {
        "model_name": model_cfg["name"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_source": "manual" if args.checkpoint is not None else "model_zoo",
        "dataset_root": str(Path(data_cfg["root"]).resolve()),
        "split": args.split,
        "top1": float(eval_metrics["top1"]),
        "top5": float(eval_metrics["top5"]),
        "num_samples": int(eval_metrics["num_samples"]),
        "seed": seed,
        "deterministic": deterministic,
        "batch_size": int(data_cfg["batch_size"]),
        "img_size": int(data_cfg["img_size"]),
        "device": str(device),
        "eval_path": str(run_paths["eval_path"]),
    }
    dump_json(run_paths["eval_path"], result)
    _print_result(result, run_paths["eval_path"])


if __name__ == "__main__":
    main()
