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

from data.build_loader import build_eval_loader, get_class_names
from engine.evaluator import evaluate
from models.baseline_models import build_model_from_cfg
from utils.artifacts import build_run_paths, dump_csv, dump_json
from utils.model_zoo import resolve_model_zoo_best_checkpoint
from utils.reproducibility import seed_everything

PREDICTION_FIELDS = [
    "image",
    "top1_index",
    "top1_class",
    "top1_prob",
    "top5_indices",
    "top5_classes",
    "top5_probs",
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
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional path to a specific checkpoint. If omitted, use results/models/<model_name>/best.pt.",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="train/val compute metrics; test runs unlabeled inference and exports predictions.",
    )
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


def _print_eval_result(result: dict[str, Any], eval_path: Path) -> None:
    print("=" * 80)
    print("Tiny-ImageNet-200 | DeiT-Tiny validation evaluation")
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


def _print_test_result(result: dict[str, Any], eval_path: Path, predictions_path: Path) -> None:
    print("=" * 80)
    print("Tiny-ImageNet-200 | DeiT-Tiny test inference")
    print("=" * 80)
    print(f"model_name       : {result['model_name']}")
    print(f"checkpoint       : {result['checkpoint_path']}")
    print(f"checkpoint src   : {result['checkpoint_source']}")
    print(f"dataset_root     : {result['dataset_root']}")
    print(f"split            : {result['split']}")
    print(f"num_samples      : {result['num_samples']}")
    print(f"class_count      : {result['num_classes']}")
    print(f"predictions file : {predictions_path}")
    print(f"summary file     : {eval_path}")
    print("=" * 80)


def _serialize_indices(values: list[int]) -> str:
    return "|".join(str(value) for value in values)


def _serialize_strings(values: list[str]) -> str:
    return "|".join(values)


def _serialize_probs(values: list[float]) -> str:
    return "|".join(f"{value:.6f}" for value in values)


def _predict_test_split(model, loader, device, class_names: list[str]) -> dict[str, Any]:
    model.eval()
    rows: list[dict[str, Any]] = []
    total = 0

    with torch.no_grad():
        for images, image_names in loader:
            images = images.to(device, non_blocking=True)
            probs = torch.softmax(model(images), dim=1)
            max_k = min(5, probs.size(1))
            top_probs, top_indices = probs.topk(max_k, dim=1)
            top_probs = top_probs.cpu()
            top_indices = top_indices.cpu()

            for image_name, sample_indices, sample_probs in zip(image_names, top_indices, top_probs):
                indices_list = [int(value) for value in sample_indices.tolist()]
                probs_list = [float(value) for value in sample_probs.tolist()]
                classes_list = [class_names[index] for index in indices_list]
                rows.append(
                    {
                        "image": image_name,
                        "top1_index": indices_list[0],
                        "top1_class": classes_list[0],
                        "top1_prob": probs_list[0],
                        "top5_indices": _serialize_indices(indices_list),
                        "top5_classes": _serialize_strings(classes_list),
                        "top5_probs": _serialize_probs(probs_list),
                    }
                )
                total += 1

    return {
        "num_samples": total,
        "rows": rows,
    }


def _build_predictions_path(run_paths: dict[str, Path], config_stem: str, split: str) -> Path:
    return (
        ROOT
        / "results"
        / "predictions"
        / run_paths["date_str"]
        / f"{config_stem}_{run_paths['run_id']}_{split}.csv"
    )


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
        checkpoint_source = "manual"
    else:
        checkpoint_path, _ = resolve_model_zoo_best_checkpoint(ROOT / "results", model_cfg["name"])
        checkpoint_source = "model_zoo"

    checkpoint = _load_checkpoint(checkpoint_path)
    _require_keys(checkpoint, ["model_state"], f"checkpoint {checkpoint_path}")

    model = build_model_from_cfg(model_cfg, pretrained_override=False).to(device)

    try:
        model.load_state_dict(checkpoint["model_state"])
    except Exception as exc:
        raise RuntimeError(f"Failed to load model weights from {checkpoint_path}: {exc}") from exc

    config_stem = Path(args.config).stem
    run_paths = build_run_paths(ROOT / "results", config_stem, eval_split=args.split)
    run_paths["eval_path"].parent.mkdir(parents=True, exist_ok=True)

    common_result = {
        "model_name": model_cfg["name"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_source": checkpoint_source,
        "dataset_root": str(Path(data_cfg["root"]).resolve()),
        "split": args.split,
        "seed": seed,
        "deterministic": deterministic,
        "batch_size": int(data_cfg["batch_size"]),
        "img_size": int(data_cfg["img_size"]),
        "device": str(device),
    }

    if args.split == "test":
        class_names = get_class_names(cfg)
        test_loader, _ = build_eval_loader(cfg, split="test")
        predictions = _predict_test_split(model, test_loader, device, class_names)
        predictions_path = _build_predictions_path(run_paths, config_stem, args.split)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        dump_csv(predictions_path, predictions["rows"], PREDICTION_FIELDS)

        result = {
            **common_result,
            "mode": "test_inference",
            "num_samples": int(predictions["num_samples"]),
            "num_classes": len(class_names),
            "predictions_path": str(predictions_path),
            "eval_path": str(run_paths["eval_path"]),
        }
        dump_json(run_paths["eval_path"], result)
        _print_test_result(result, run_paths["eval_path"], predictions_path)
        return

    eval_loader, _ = build_eval_loader(cfg, split=args.split)
    eval_metrics = evaluate(model, eval_loader, device)
    result = {
        **common_result,
        "mode": "labeled_evaluation",
        "top1": float(eval_metrics["top1"]),
        "top5": float(eval_metrics["top5"]),
        "num_samples": int(eval_metrics["num_samples"]),
        "eval_path": str(run_paths["eval_path"]),
    }
    dump_json(run_paths["eval_path"], result)
    _print_eval_result(result, run_paths["eval_path"])


if __name__ == "__main__":
    main()
