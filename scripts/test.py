from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.build_loader import build_eval_loader, get_class_names
from engine.evaluator import evaluate
from models import build_model_from_cfg
from utils.artifacts import build_run_paths, dump_csv, dump_json
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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    return parser.parse_args()


def _load_checkpoint(path: Path):
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Checkpoint not found: {path}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint from {path}: {exc}") from exc
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(f"Checkpoint at {path} must contain model_state")
    return checkpoint


def _serialize_indices(values: list[int]) -> str:
    return "|".join(str(value) for value in values)


def _serialize_strings(values: list[str]) -> str:
    return "|".join(values)


def _serialize_probs(values: list[float]) -> str:
    return "|".join(f"{value:.6f}" for value in values)


def _predict_test_split(model, loader, device, class_names: list[str]):
    model.eval()
    rows = []
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

    return {"num_samples": total, "rows": rows}


def main():
    args = parse_args()

    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(config_path)

    seed = int(_get(cfg, "train", "seed", default=42))
    deterministic = bool(_get(cfg, "train", "deterministic", default=True))
    seed_everything(seed, deterministic=deterministic)

    device_cfg = _get(cfg, "train", "device", default="cpu")
    device = torch.device("cuda" if device_cfg == "cuda" and torch.cuda.is_available() else "cpu")

    checkpoint = _load_checkpoint(checkpoint_path)
    model = build_model_from_cfg(cfg["model"], pretrained_override=False).to(device)
    model.load_state_dict(checkpoint["model_state"])

    run_paths = build_run_paths(ROOT / "results", config_path.stem, eval_split=args.split)
    eval_path = Path(run_paths["eval_path"])
    eval_path.parent.mkdir(parents=True, exist_ok=True)

    common_result = {
        "model_name": cfg["model"]["name"],
        "checkpoint_path": str(checkpoint_path),
        "model_source": str(checkpoint.get("model_state_source", "model")),
        "dataset_root": str(Path(cfg["data"]["root"]).resolve()),
        "split": args.split,
        "batch_size": int(cfg["data"]["batch_size"]),
        "img_size": int(cfg["data"]["img_size"]),
        "device": str(device),
    }

    if args.split == "test":
        class_names = get_class_names(cfg)
        loader, _ = build_eval_loader(cfg, split="test")
        predictions = _predict_test_split(model, loader, device, class_names)
        predictions_path = (
            ROOT
            / "results"
            / "predictions"
            / str(run_paths["date_str"])
            / f"{config_path.stem}_{run_paths['run_id']}_test.csv"
        )
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        dump_csv(predictions_path, predictions["rows"], PREDICTION_FIELDS)

        result = {
            **common_result,
            "mode": "test_inference",
            "num_samples": int(predictions["num_samples"]),
            "num_classes": len(class_names),
            "predictions_path": str(predictions_path),
        }
        dump_json(eval_path, result)
        print("=" * 80)
        print("Tiny-ImageNet | DeiT-Tiny test inference")
        print("=" * 80)
        print(f"checkpoint : {checkpoint_path}")
        print(f"source     : {common_result['model_source']}")
        print(f"predictions: {predictions_path}")
        print(f"summary    : {eval_path}")
        return

    loader, _ = build_eval_loader(cfg, split=args.split)
    metrics = evaluate(model, loader, device)
    result = {
        **common_result,
        "mode": "labeled_evaluation",
        "top1": float(metrics["top1"]),
        "top5": float(metrics["top5"]),
        "num_samples": int(metrics["num_samples"]),
    }
    dump_json(eval_path, result)

    print("=" * 80)
    print("Tiny-ImageNet | DeiT-Tiny evaluation")
    print("=" * 80)
    print(f"checkpoint : {checkpoint_path}")
    print(f"source     : {result['model_source']}")
    print(f"split      : {args.split}")
    print(f"top1       : {result['top1']:.2f}%")
    print(f"top5       : {result['top5']:.2f}%")
    print(f"eval file  : {eval_path}")


if __name__ == "__main__":
    main()
