from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from utils.artifacts import build_model_zoo_paths, dump_json


def _get_score(data: dict[str, Any]) -> float | None:
    for key in ("best_val_acc", "eval_top1", "best_acc", "top1", "val_acc"):
        value = data.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _load_existing_record(paths: dict[str, Path]) -> tuple[dict[str, Any] | None, float | None]:
    if paths["best_metadata_path"].exists():
        with open(paths["best_metadata_path"], "r", encoding="utf-8") as f:
            record = json.load(f)
        score = _get_score(record)
        if score is not None:
            return record, score

    if paths["best_checkpoint_path"].exists():
        checkpoint = torch.load(paths["best_checkpoint_path"], map_location="cpu")
        if isinstance(checkpoint, dict):
            score = _get_score(checkpoint)
            if score is not None:
                record = {
                    "model_name": checkpoint.get("model_name"),
                    "best_val_acc": score,
                    "best_checkpoint_path": str(paths["best_checkpoint_path"]),
                    "source_checkpoint_path": str(paths["best_checkpoint_path"]),
                    "source_run_id": checkpoint.get("run_id"),
                    "source_config_path": checkpoint.get("config_path"),
                    "source_summary_path": checkpoint.get("summary_path"),
                    "source_metrics_path": checkpoint.get("metrics_path"),
                    "source_eval_path": checkpoint.get("eval_result_path"),
                    "source_dataset": checkpoint.get("dataset"),
                    "source_img_size": checkpoint.get("img_size"),
                    "source_batch_size": checkpoint.get("batch_size"),
                    "updated_at": None,
                    "score_key": "checkpoint",
                }
                return record, score

    return None, None


def sync_model_zoo_best(
    results_root: Path,
    model_name: str,
    source_checkpoint_path: Path,
    source_record: dict[str, Any],
) -> dict[str, Any]:
    paths = build_model_zoo_paths(results_root, model_name)
    existing_record, existing_score = _load_existing_record(paths)
    current_score = _get_score(source_record)
    if current_score is None:
        raise ValueError("source_record must contain one of: best_val_acc, eval_top1, best_acc, top1, val_acc")

    updated = existing_score is None or current_score > existing_score
    model_dir = paths["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)

    if updated:
        shutil.copy2(source_checkpoint_path, paths["best_checkpoint_path"])
        metadata = {
            "model_name": model_name,
            "best_val_acc": current_score,
            "score_key": "best_val_acc" if "best_val_acc" in source_record else "eval_top1",
            "best_checkpoint_path": str(paths["best_checkpoint_path"]),
            "source_checkpoint_path": str(source_checkpoint_path),
            "source_run_id": source_record.get("run_id"),
            "source_date_str": source_record.get("date_str"),
            "source_config_path": source_record.get("config_path"),
            "source_summary_path": source_record.get("summary_path"),
            "source_metrics_path": source_record.get("metrics_path"),
            "source_eval_path": source_record.get("eval_path"),
            "source_dataset": source_record.get("dataset"),
            "source_img_size": source_record.get("img_size"),
            "source_batch_size": source_record.get("batch_size"),
            "source_epochs_completed": source_record.get("epochs_completed"),
            "source_eval_top1": source_record.get("eval_top1"),
            "source_eval_top5": source_record.get("eval_top5"),
            "source_checkpoint_type": source_record.get("type"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        dump_json(paths["best_metadata_path"], metadata)
        return {
            "updated": True,
            "existing_score": existing_score,
            "best_score": current_score,
            "best_checkpoint_path": paths["best_checkpoint_path"],
            "best_metadata_path": paths["best_metadata_path"],
            "record": metadata,
            "paths": paths,
        }

    if existing_record is not None and not paths["best_metadata_path"].exists():
        metadata = dict(existing_record)
        metadata["best_checkpoint_path"] = str(paths["best_checkpoint_path"])
        dump_json(paths["best_metadata_path"], metadata)
        existing_record = metadata

    return {
        "updated": False,
        "existing_score": existing_score,
        "best_score": existing_score,
        "best_checkpoint_path": paths["best_checkpoint_path"],
        "best_metadata_path": paths["best_metadata_path"],
        "record": existing_record,
        "paths": paths,
    }


def resolve_model_zoo_best_checkpoint(results_root: Path, model_name: str) -> tuple[Path, dict[str, Any] | None]:
    paths = build_model_zoo_paths(results_root, model_name)
    if paths["best_checkpoint_path"].exists():
        metadata = None
        if paths["best_metadata_path"].exists():
            with open(paths["best_metadata_path"], "r", encoding="utf-8") as f:
                metadata = json.load(f)
        return paths["best_checkpoint_path"], metadata

    if paths["best_metadata_path"].exists():
        with open(paths["best_metadata_path"], "r", encoding="utf-8") as f:
            metadata = json.load(f)
        for key in ("best_checkpoint_path", "checkpoint_path", "source_checkpoint_path"):
            candidate = metadata.get(key)
            if candidate:
                candidate_path = Path(candidate)
                if candidate_path.exists():
                    return candidate_path, metadata

    raise FileNotFoundError(
        f"No model-zoo best checkpoint found for model {model_name!r}. "
        f"Expected {paths['best_checkpoint_path']}. Train the model first."
    )
