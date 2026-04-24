#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


@dataclass(frozen=True)
class Experiment:
    key: str
    label: str
    summary_path: Path
    metrics_path: Path
    note: str = ""


EXPERIMENTS: dict[str, Experiment] = {
    "baseline_224": Experiment(
        key="baseline_224",
        label="224 baseline",
        summary_path=ROOT / "results/summary/20260422/deit_tiny_baseline_20260422_233858.md",
        metrics_path=ROOT / "results/metrics/20260422/deit_tiny_baseline_20260422_233858.csv",
        note="timm deit_tiny_patch16_224",
    ),
    "baseline_112": Experiment(
        key="baseline_112",
        label="112 baseline",
        summary_path=ROOT / "results/summary/20260421/deit_tiny_patch8_112_baseline_20260421_003247.md",
        metrics_path=ROOT / "results/metrics/20260421/deit_tiny_patch8_112_baseline_20260421_003247.csv",
        note="patch8, 150 epochs",
    ),
    "baseline_112_300": Experiment(
        key="baseline_112_300",
        label="112 baseline (300 ep)",
        summary_path=ROOT / "results/summary/20260422/deit_tiny_patch8_112_baseline_300ep_20260422_181902.md",
        metrics_path=ROOT / "results/metrics/20260422/deit_tiny_patch8_112_baseline_300ep_20260422_181902.csv",
        note="patch8, 300 epochs",
    ),
    "prepatch": Experiment(
        key="prepatch",
        label="PrePatch",
        summary_path=ROOT / "results/summary/20260421/deit_tiny_patch8_112_prepatch_20260421_160846.md",
        metrics_path=ROOT / "results/metrics/20260421/deit_tiny_patch8_112_prepatch_20260421_160846.csv",
    ),
    "precnn": Experiment(
        key="precnn",
        label="PreCNN",
        summary_path=ROOT / "results/summary/20260421/deit_tiny_patch8_112_precnn_20260421_143914.md",
        metrics_path=ROOT / "results/metrics/20260421/deit_tiny_patch8_112_precnn_20260421_143914.csv",
    ),
    "localffn": Experiment(
        key="localffn",
        label="Local-FFN",
        summary_path=ROOT / "results/summary/20260421/deit_tiny_patch8_112_localffn_20260421_143442.md",
        metrics_path=ROOT / "results/metrics/20260421/deit_tiny_patch8_112_localffn_20260421_143442.csv",
    ),
    "overlap_patch12": Experiment(
        key="overlap_patch12",
        label="Overlap patch12",
        summary_path=ROOT / "results/summary/20260422/deit_tiny_patch8_112_overlap_patch12_20260422_000455.md",
        metrics_path=ROOT / "results/metrics/20260422/deit_tiny_patch8_112_overlap_patch12_20260422_000455.csv",
        note="kernel=12, stride=8",
    ),
    "strong_teacher": Experiment(
        key="strong_teacher",
        label="Teacher (strong)",
        summary_path=ROOT / "results/summary/20260422/deit_tiny_patch8_112_overlap_patch12_strong_teacher_20260422_125604.md",
        metrics_path=ROOT / "results/metrics/20260422/deit_tiny_patch8_112_overlap_patch12_strong_teacher_20260422_125604.csv",
    ),
    "teacher_polish40": Experiment(
        key="teacher_polish40",
        label="Teacher + polish40",
        summary_path=ROOT / "results/summary/20260422/deit_tiny_patch8_112_overlap_patch12_strong_teacher_polish40_20260422_171852.md",
        metrics_path=ROOT / "results/metrics/20260422/deit_tiny_patch8_112_overlap_patch12_strong_teacher_polish40_20260422_171852.csv",
    ),
    "teacher_twostage": Experiment(
        key="teacher_twostage",
        label="Teacher (final)",
        summary_path=ROOT / "results/summary/20260423/deit_tiny_patch8_112_overlap_patch12_teacher_twostage_20260423_232954.md",
        metrics_path=ROOT / "results/metrics/20260423/deit_tiny_patch8_112_overlap_patch12_teacher_twostage_20260423_232954.csv",
        note="two-stage refinement",
    ),
    "student_d9_soft": Experiment(
        key="student_d9_soft",
        label="Student d9 soft-KD",
        summary_path=ROOT / "results/summary/20260424/deit_tiny_patch8_112_student_depth9_logit_softkd_20260424_131100.md",
        metrics_path=ROOT / "results/metrics/20260424/deit_tiny_patch8_112_student_depth9_logit_softkd_20260424_131100.csv",
    ),
    "student_d9_hard": Experiment(
        key="student_d9_hard",
        label="Student d9 hard-KD",
        summary_path=ROOT / "results/summary/20260424/deit_tiny_patch8_112_student_depth9_deit_harddistill_same_recipe_20260424_170621.md",
        metrics_path=ROOT / "results/metrics/20260424/deit_tiny_patch8_112_student_depth9_deit_harddistill_same_recipe_20260424_170621.csv",
    ),
    "student_d10_soft": Experiment(
        key="student_d10_soft",
        label="Student d10 soft-KD",
        summary_path=ROOT / "results/summary/20260423/deit_tiny_patch8_112_student_depth10_logit_softkd_20260423_221731.md",
        metrics_path=ROOT / "results/metrics/20260423/deit_tiny_patch8_112_student_depth10_logit_softkd_20260423_221731.csv",
    ),
    "student_d10_hard": Experiment(
        key="student_d10_hard",
        label="Student d10 hard-KD",
        summary_path=ROOT / "results/summary/20260424/deit_tiny_patch8_112_student_depth10_deit_harddistill_same_recipe_20260424_133243.md",
        metrics_path=ROOT / "results/metrics/20260424/deit_tiny_patch8_112_student_depth10_deit_harddistill_same_recipe_20260424_133243.csv",
    ),
    "student_d10_final": Experiment(
        key="student_d10_final",
        label="Student d10 final",
        summary_path=ROOT / "results/summary/20260424/deit_tiny_patch8_112_student_depth10_deit_harddistill_twostage_20260424_180111.md",
        metrics_path=ROOT / "results/metrics/20260424/deit_tiny_patch8_112_student_depth10_deit_harddistill_twostage_20260424_180111.csv",
        note="hard-KD + two-stage refinement",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate thesis-ready curves and tables.")
    default_dir = ROOT / "results" / "figures" / datetime.now().strftime("%Y%m%d") / "thesis_visuals"
    parser.add_argument("--output-dir", type=Path, default=default_dir)
    parser.add_argument("--png-dpi", type=int, default=400)
    return parser.parse_args()


def read_metrics(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row: dict[str, float] = {}
            for key, value in raw.items():
                if value is None or value == "":
                    continue
                row[key] = float(value)
            rows.append(row)
    return rows


def read_summary(path: Path) -> dict[str, str]:
    summary: dict[str, str] = {}
    pattern = re.compile(r"^- ([^:]+):\s*(.*)$")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = pattern.match(line.strip())
            if match:
                summary[match.group(1).strip()] = match.group(2).strip()
    return summary


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def best_top1(metrics: list[dict[str, float]]) -> float:
    return max(row["val_acc"] for row in metrics)


def best_epoch(metrics: list[dict[str, float]]) -> int:
    best_row = max(metrics, key=lambda row: row["val_acc"])
    return int(best_row["epoch"])


def nice_ylim(curves: Iterable[list[dict[str, float]]]) -> tuple[float, float]:
    values = [row["val_acc"] for curve in curves for row in curve]
    lower = min(values)
    upper = max(values)
    low = max(0.0, math.floor((lower - 1.0) / 5.0) * 5.0)
    high = min(100.0, math.ceil((upper + 1.0) / 5.0) * 5.0)
    return low, high


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, png_dpi: int) -> tuple[Path, Path]:
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=png_dpi, bbox_inches="tight")
    return pdf_path, png_path


def plot_curve_panel(
    experiment_keys: list[str],
    title: str,
    output_dir: Path,
    stem: str,
    png_dpi: int,
    stage_line_epoch: int | None = None,
) -> plt.Figure:
    palette = plt.get_cmap("tab10")
    metrics_map = {key: read_metrics(EXPERIMENTS[key].metrics_path) for key in experiment_keys}
    fig, ax = plt.subplots(figsize=(11.5, 6.4))

    for idx, key in enumerate(experiment_keys):
        exp = EXPERIMENTS[key]
        rows = metrics_map[key]
        epochs = [int(row["epoch"]) for row in rows]
        val_acc = [row["val_acc"] for row in rows]
        color = palette(idx % 10)
        ax.plot(
            epochs,
            val_acc,
            label=f"{exp.label} ({best_top1(rows):.2f}%)",
            color=color,
            linewidth=2.2,
        )
        best_idx = max(range(len(rows)), key=lambda i: rows[i]["val_acc"])
        ax.scatter(
            epochs[best_idx],
            val_acc[best_idx],
            color=color,
            s=34,
            zorder=3,
            edgecolors="white",
            linewidths=0.8,
        )

    low, high = nice_ylim(metrics_map.values())
    ax.set_ylim(low, high)
    if stage_line_epoch is not None:
        ax.axvline(stage_line_epoch, color="#444444", linestyle="--", linewidth=1.2)
        ax.text(
            stage_line_epoch + 1,
            high - 0.6,
            "refine stage",
            fontsize=10,
            color="#444444",
            va="top",
        )
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Top-1 (%)")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)

    save_figure(fig, output_dir, stem, png_dpi)
    return fig


def to_float(summary: dict[str, str], key: str) -> float:
    return float(summary[key])


def build_table_rows(keys: list[str], extra_columns: list[tuple[str, callable]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for key in keys:
        exp = EXPERIMENTS[key]
        summary = read_summary(exp.summary_path)
        row = [
            exp.label,
            summary.get("img_size", "N/A"),
            f"{to_float(summary, 'best_val_acc'):.2f}",
            f"{to_float(summary, 'params_m'):.2f}",
            f"{to_float(summary, 'flops_g'):.2f}",
        ]
        for _, func in extra_columns:
            row.append(func(exp, summary))
        rows.append(row)
    return rows


def write_csv_table(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def render_table_figure(
    title: str,
    headers: list[str],
    rows: list[list[str]],
    output_dir: Path,
    stem: str,
    png_dpi: int,
    highlight_row_name: str | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12.2, 1.8 + 0.52 * len(rows)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        bbox=[0.0, 0.02, 1.0, 0.82],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.32)

    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        cell.set_edgecolor("#9aa4ad")
        if r == 0:
            cell.set_facecolor("#dfe8f3")
            cell.set_text_props(weight="bold", color="#1f2933")
        else:
            cell.set_facecolor("#f9fbfd" if r % 2 == 1 else "#eef3f7")
            if highlight_row_name is not None and rows[r - 1][0] == highlight_row_name:
                cell.set_facecolor("#d8efe0")
                cell.set_text_props(weight="bold")

    ax.set_title(title, pad=8)
    fig.subplots_adjust(left=0.015, right=0.985, top=0.88, bottom=0.04)
    save_figure(fig, output_dir, stem, png_dpi)
    return fig


def render_readme(output_dir: Path, exported_files: list[Path]) -> None:
    lines = [
        "# Thesis Visual Outputs",
        "",
        "本目录包含可直接用于毕业论文的高分辨率曲线图、对比表和合并 PDF。",
        "",
        "## 关于 train loss",
        "",
        "- 本次默认不单独输出 train loss 图。",
        "- 原因是当前主线训练广泛使用了 mixup、cutmix、label smoothing 与 EMA。",
        "- 在这种设置下，train loss 不再对应传统 one-hot 监督下的“是否过拟合”，数值也不会像常规训练那样下降到很低。",
        "- 因此论文主体更建议使用 val acc 曲线来展示收敛速度、稳定性和模型间差异。",
        "",
        "## 输出文件",
        "",
    ]
    for file in sorted(exported_files):
        lines.append(f"- `{file.name}`")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)
    exported_files: list[Path] = []

    figures: list[plt.Figure] = []

    figures.append(
        plot_curve_panel(
            ["baseline_224", "baseline_112", "teacher_twostage", "student_d10_final"],
            title="Validation Curves: Baselines, Final Teacher and Final Student",
            output_dir=output_dir,
            stem="curve_mainline_overview",
            png_dpi=args.png_dpi,
            stage_line_epoch=150,
        )
    )
    figures.append(
        plot_curve_panel(
            ["baseline_112", "prepatch", "precnn", "localffn", "overlap_patch12"],
            title="Validation Curves: 112 Mainline Structure Ablation",
            output_dir=output_dir,
            stem="curve_structure_ablation_112",
            png_dpi=args.png_dpi,
        )
    )
    figures.append(
        plot_curve_panel(
            ["overlap_patch12", "strong_teacher", "teacher_polish40", "teacher_twostage"],
            title="Validation Curves: Teacher Evolution on 112 Mainline",
            output_dir=output_dir,
            stem="curve_teacher_evolution",
            png_dpi=args.png_dpi,
            stage_line_epoch=150,
        )
    )
    figures.append(
        plot_curve_panel(
            ["student_d9_soft", "student_d9_hard", "student_d10_soft", "student_d10_hard", "student_d10_final"],
            title="Validation Curves: Final Student Selection",
            output_dir=output_dir,
            stem="curve_student_selection",
            png_dpi=args.png_dpi,
            stage_line_epoch=150,
        )
    )

    main_headers = ["Setting", "Input", "Top-1 (%)", "Params (M)", "FLOPs (G)", "Remark"]
    main_rows = build_table_rows(
        ["baseline_224", "baseline_112", "teacher_twostage", "student_d10_final"],
        [("Remark", lambda exp, summary: exp.note or "-")],
    )
    write_csv_table(output_dir / "table_main_results.csv", main_headers, main_rows)
    figures.append(
        render_table_figure(
            "Main Comparison: Baseline, Teacher and Final Student",
            headers=main_headers,
            rows=main_rows,
            output_dir=output_dir,
            stem="table_main_results",
            png_dpi=args.png_dpi,
            highlight_row_name="Student d10 final",
        )
    )

    structure_base = read_summary(EXPERIMENTS["baseline_112"].summary_path)
    base_acc = to_float(structure_base, "best_val_acc")
    structure_headers = ["Model", "Input", "Top-1 (%)", "Params (M)", "FLOPs (G)", "Delta vs 112 base"]
    structure_rows = build_table_rows(
        ["baseline_112", "prepatch", "precnn", "localffn", "overlap_patch12"],
        [
            (
                "Delta",
                lambda exp, summary: f"{to_float(summary, 'best_val_acc') - base_acc:+.2f}",
            )
        ],
    )
    write_csv_table(output_dir / "table_structure_ablation_112.csv", structure_headers, structure_rows)
    figures.append(
        render_table_figure(
            "Structure Ablation on 112 Mainline",
            headers=structure_headers,
            rows=structure_rows,
            output_dir=output_dir,
            stem="table_structure_ablation_112",
            png_dpi=args.png_dpi,
            highlight_row_name="Overlap patch12",
        )
    )

    teacher_base = read_summary(EXPERIMENTS["overlap_patch12"].summary_path)
    teacher_acc = to_float(teacher_base, "best_val_acc")
    teacher_headers = ["Teacher Stage", "Input", "Top-1 (%)", "Params (M)", "FLOPs (G)", "Delta vs overlap"]
    teacher_rows = build_table_rows(
        ["overlap_patch12", "strong_teacher", "teacher_polish40", "teacher_twostage"],
        [
            (
                "Delta",
                lambda exp, summary: f"{to_float(summary, 'best_val_acc') - teacher_acc:+.2f}",
            )
        ],
    )
    write_csv_table(output_dir / "table_teacher_evolution.csv", teacher_headers, teacher_rows)
    figures.append(
        render_table_figure(
            "Teacher Evolution on 112 Mainline",
            headers=teacher_headers,
            rows=teacher_rows,
            output_dir=output_dir,
            stem="table_teacher_evolution",
            png_dpi=args.png_dpi,
            highlight_row_name="Teacher (final)",
        )
    )

    student_base = read_summary(EXPERIMENTS["student_d10_final"].summary_path)
    student_acc = to_float(student_base, "best_val_acc")
    student_headers = ["Candidate", "Input", "Top-1 (%)", "Params (M)", "FLOPs (G)", "Delta vs final"]
    student_rows = build_table_rows(
        ["student_d9_soft", "student_d9_hard", "student_d10_soft", "student_d10_hard", "student_d10_final"],
        [
            (
                "Delta",
                lambda exp, summary: f"{to_float(summary, 'best_val_acc') - student_acc:+.2f}",
            )
        ],
    )
    write_csv_table(output_dir / "table_student_selection.csv", student_headers, student_rows)
    figures.append(
        render_table_figure(
            "Final Student Selection",
            headers=student_headers,
            rows=student_rows,
            output_dir=output_dir,
            stem="table_student_selection",
            png_dpi=args.png_dpi,
            highlight_row_name="Student d10 final",
        )
    )

    for stem in [
        "curve_mainline_overview",
        "curve_structure_ablation_112",
        "curve_teacher_evolution",
        "curve_student_selection",
        "table_main_results",
        "table_structure_ablation_112",
        "table_teacher_evolution",
        "table_student_selection",
    ]:
        exported_files.append(output_dir / f"{stem}.pdf")
        exported_files.append(output_dir / f"{stem}.png")
    exported_files.extend(
        [
            output_dir / "table_main_results.csv",
            output_dir / "table_structure_ablation_112.csv",
            output_dir / "table_teacher_evolution.csv",
            output_dir / "table_student_selection.csv",
        ]
    )

    bundle_path = output_dir / "thesis_visuals_bundle.pdf"
    with PdfPages(bundle_path) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches="tight")
    exported_files.append(bundle_path)

    render_readme(output_dir, exported_files)
    exported_files.append(output_dir / "README.md")

    for fig in figures:
        plt.close(fig)

    print(f"Generated thesis visuals in: {output_dir}")
    print(f"Bundle PDF: {bundle_path}")


if __name__ == "__main__":
    main()
