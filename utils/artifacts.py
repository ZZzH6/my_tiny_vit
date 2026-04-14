from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def build_run_paths(
    results_root: Path,
    config_stem: str,
    run_id: str | None = None,
    eval_split: str = "val",
) -> dict[str, Path | str]:
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    date_str = run_id[:8]

    return {
        "run_id": run_id,
        "date_str": date_str,
        "log_path": results_root / "logs" / date_str / f"{config_stem}_{run_id}.log",
        "metrics_path": results_root / "metrics" / date_str / f"{config_stem}_{run_id}.csv",
        "summary_path": results_root / "summary" / date_str / f"{config_stem}_{run_id}.md",
        "eval_path": results_root / "eval" / date_str / f"{config_stem}_{run_id}_{eval_split}.json",
        "best_checkpoint_path": results_root / "checkpoints" / date_str / f"{config_stem}_{run_id}_best.pt",
        "last_checkpoint_path": results_root / "checkpoints" / date_str / f"{config_stem}_{run_id}_last.pt",
    }


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, data: Any) -> Path:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def dump_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> Path:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path
