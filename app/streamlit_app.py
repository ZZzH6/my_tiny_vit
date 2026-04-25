from __future__ import annotations

import io
import json
from pathlib import Path
from textwrap import dedent

from html import escape

import matplotlib.pyplot as plt
import streamlit as st
import torch
from PIL import Image

from system_runtime import (
    CheckpointLoadError,
    MODEL_SPECS,
    build_model_catalog,
    build_scatter_frame,
    get_model_meta,
    latest_thesis_visual_dir,
    load_csv_if_exists,
    predict_batch,
    predict_image,
)


st.set_page_config(
    page_title="TinyViT System Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


SAMPLE_IMAGE_PATH = Path(__file__).resolve().parents[1] / "results/resize_demo/dog64x64_deit_bicubic_112.jpg"
DEPLOYMENT_METADATA_PATH = Path(__file__).with_name("deployment_metadata.json")
OVERVIEW_STATIC_FALLBACKS = {
    "baseline_112": {
        "best_val_acc": 79.46,
        "params_m": 5.45,
        "flops_g": 2.106043392,
        "input_size": 112,
    },
    "teacher_final": {
        "best_val_acc": 80.18,
        "params_m": 5.50,
        "flops_g": 2.124106752,
        "input_size": 112,
    },
}


@st.cache_data(show_spinner=False)
def load_deployment_metadata() -> dict[str, object]:
    if not DEPLOYMENT_METADATA_PATH.exists():
        return {}
    try:
        with DEPLOYMENT_METADATA_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def safe_float(value: object) -> float | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def merge_overview_metadata(model_key: str) -> dict[str, object]:
    meta = get_model_meta(model_key)
    summary = meta.get("summary", {})
    merged: dict[str, object] = {}
    merged.update(OVERVIEW_STATIC_FALLBACKS.get(model_key, {}))
    if isinstance(summary, dict):
        merged.update(summary)
    if model_key == "student_final":
        merged.update(load_deployment_metadata())
    return merged


def format_overview_metric(value: float | None, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}{suffix}"


def compute_flop_reduction(student_flops: float | None, baseline_flops: float | None) -> float | None:
    if student_flops is None or baseline_flops in (None, 0.0):
        return None
    return (1.0 - student_flops / baseline_flops) * 100.0


def compute_teacher_gap(student_top1: float | None, teacher_top1: float | None) -> float | None:
    if student_top1 is None or teacher_top1 is None:
        return None
    return teacher_top1 - student_top1


def normalize_html_fragment(fragment: str) -> str:
    lines = dedent(fragment).splitlines()
    return "\n".join(line.strip() for line in lines if line.strip())


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f5f4ed;
            --surface: #faf9f5;
            --surface-soft: #ffffff;
            --surface-warm: #f0eee6;
            --text: #141413;
            --muted: #5e5d59;
            --muted-soft: #87867f;
            --line: #e8e6dc;
            --accent: #c96442;
            --accent-soft: rgba(201, 100, 66, 0.10);
            --accent-deep: #b95736;
            --focus: #3898ec;
            --shadow: 0 4px 24px rgba(20, 20, 19, 0.05);
            --ring: 0 0 0 1px #f0eee6;
        }

        html {
            color-scheme: light;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(201, 100, 66, 0.07), transparent 24%),
                linear-gradient(180deg, #f7f6f0 0%, #f5f4ed 100%);
            color: var(--text);
        }

        html, body, [class*="css"] {
            font-family: "Aptos", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
            color: var(--text);
        }

        body,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"],
        [data-testid="stHeader"] {
            background: transparent !important;
            color: var(--text);
        }

        .stApp p,
        .stApp li,
        .stApp label,
        .stApp small,
        .stApp span,
        .stApp .stCaption,
        .stApp [data-testid="stMarkdownContainer"],
        .stApp [data-testid="stMarkdownContainer"] p,
        .stApp [data-testid="stMarkdownContainer"] li,
        .stApp [data-testid="stWidgetLabel"],
        .stApp [data-testid="stWidgetLabel"] * {
            color: var(--text) !important;
        }

        .stApp input,
        .stApp textarea,
        .stApp select,
        .stApp option {
            color: var(--text) !important;
        }

        .stApp ::placeholder {
            color: var(--muted-soft) !important;
        }

        code, pre, .mono {
            font-family: "IBM Plex Mono", "JetBrains Mono", monospace !important;
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 3.4rem;
            max-width: 1280px;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f7f5ef, #f3f1ea);
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] * {
            color: var(--text);
        }

        #MainMenu,
        [data-testid="stToolbar"] {
            display: none !important;
        }

        [data-testid="stMetricValue"] {
            color: var(--text);
        }

        [data-testid="stMetricLabel"] {
            color: var(--muted);
        }

        [data-baseweb="select"] > div,
        [data-baseweb="base-input"] > div,
        .stTextInput input,
        .stNumberInput input {
            background: var(--surface-soft) !important;
            color: var(--text) !important;
            border-color: var(--line) !important;
        }

        [data-baseweb="select"] *,
        [data-baseweb="base-input"] * {
            color: var(--text) !important;
        }

        [data-baseweb="select"] svg,
        [data-baseweb="base-input"] svg {
            color: var(--muted) !important;
            fill: var(--muted) !important;
        }

        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] .stCaption {
            color: var(--text);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.45rem;
            padding: 0.32rem;
            margin: 0.8rem 0 1rem 0;
            border-radius: 18px;
            border: 1px solid var(--line);
            background: var(--surface);
            box-shadow: var(--ring);
        }

        .stTabs [data-baseweb="tab"] {
            height: auto;
            padding: 0.68rem 1rem;
            border-radius: 14px;
            color: var(--muted);
            font-weight: 600;
        }

        .stTabs [aria-selected="true"] {
            background: var(--surface-soft);
            color: var(--text);
            box-shadow: var(--ring);
        }

        [data-testid="stExpander"] {
            border: 1px solid var(--line);
            border-radius: 16px;
            background: var(--surface);
            box-shadow: var(--ring);
        }

        [data-testid="stFileUploader"],
        [data-testid="stDataFrame"] {
            background: transparent;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: var(--surface-soft) !important;
            border: 1px dashed var(--line) !important;
            border-radius: 16px !important;
            box-shadow: var(--ring);
        }

        [data-testid="stFileUploaderDropzone"] * {
            color: var(--muted) !important;
        }

        [data-testid="stFileUploaderDropzone"] small,
        [data-testid="stFileUploaderDropzoneInstructions"] small {
            color: var(--muted-soft) !important;
        }

        [data-testid="stFileUploaderDropzone"] svg {
            color: var(--accent) !important;
            fill: var(--accent) !important;
        }

        [data-testid="stFileUploaderDropzone"] button {
            background: var(--surface) !important;
            color: var(--text) !important;
            border: 1px solid var(--line) !important;
        }

        [data-testid="stFileUploaderFile"] {
            background: var(--surface) !important;
            border: 1px solid var(--line) !important;
            border-radius: 12px !important;
        }

        [data-testid="stFileUploaderFile"] * {
            color: var(--text) !important;
        }

        [data-testid="stDataFrame"] [role="grid"],
        [data-testid="stDataFrame"] [data-testid="StyledDataFrameResizable"],
        [data-testid="stDataFrame"] [data-testid="stElementContainer"] {
            background: var(--surface-soft) !important;
        }

        [data-testid="stDataFrame"] * {
            color: var(--text) !important;
        }

        .stTable table,
        .stDataFrame table,
        .stTable th,
        .stTable td,
        .stDataFrame th,
        .stDataFrame td {
            background: var(--surface-soft) !important;
            color: var(--text) !important;
            border-color: var(--line) !important;
        }

        .stButton > button,
        .stDownloadButton > button {
            background: var(--surface-soft);
            color: var(--text);
            border: 1px solid var(--line);
            border-radius: 12px;
            box-shadow: var(--ring);
        }

        .hero {
            padding: 1.5rem 0 0.75rem 0;
            margin-bottom: 0.35rem;
        }

        .hero-title {
            font-family: Georgia, "Times New Roman", serif;
            font-size: 3rem;
            line-height: 1.1;
            font-weight: 500;
            letter-spacing: -0.03em;
            margin-bottom: 0.55rem;
            color: var(--text);
        }

        .hero-subtitle {
            font-size: 1rem;
            line-height: 1.75;
            max-width: 760px;
            color: var(--muted);
            margin-bottom: 0.8rem;
        }

        .chip {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.42rem 0.86rem;
            border-radius: 999px;
            border: 1px solid var(--line);
            background: var(--surface);
            color: var(--muted);
            font-size: 0.85rem;
            font-weight: 600;
            box-shadow: var(--ring);
        }

        .small-chip {
            display: inline-block;
            margin-bottom: 0.6rem;
            padding: 0.3rem 0.68rem;
            border-radius: 999px;
            background: var(--accent-soft);
            border: 1px solid rgba(201, 100, 66, 0.18);
            color: var(--accent);
            font-size: 0.76rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }

        .panel {
            padding: 1rem 1rem 0.92rem 1rem;
            border-radius: 18px;
            border: 1px solid var(--line);
            background: var(--surface);
            box-shadow: var(--ring);
        }

        .panel-title {
            font-size: 1.05rem;
            font-weight: 600;
            color: var(--text);
            margin-bottom: 0.35rem;
        }

        .panel-copy {
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.65;
        }

        .stat-card {
            padding: 1rem 1rem 0.95rem 1rem;
            border-radius: 18px;
            border: 1px solid var(--line);
            background: var(--surface);
            min-height: 142px;
            box-shadow: var(--ring), var(--shadow);
        }

        .stat-label {
            color: var(--muted);
            font-size: 0.84rem;
            margin-bottom: 0.55rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .stat-value {
            color: var(--text);
            font-size: 1.95rem;
            line-height: 1.1;
            font-weight: 600;
            letter-spacing: -0.03em;
            margin-bottom: 0.4rem;
        }

        .stat-note {
            color: var(--muted-soft);
            font-size: 0.92rem;
            line-height: 1.55;
        }

        .result-shell {
            display: flex;
            flex-direction: column;
            gap: 0.85rem;
            border-radius: 22px;
            border: 1px solid rgba(20, 20, 19, 0.08);
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.82), rgba(250, 249, 245, 0.96));
            padding: 1rem;
            min-height: 100%;
            box-shadow: var(--ring), var(--shadow);
        }

        .result-title {
            display: flex;
            justify-content: space-between;
            gap: 0.9rem;
            align-items: flex-start;
        }

        .result-name {
            color: var(--text);
            font-size: 1.14rem;
            font-weight: 600;
            line-height: 1.3;
        }

        .result-note {
            color: var(--muted-soft);
            font-size: 0.9rem;
            margin-top: 0.32rem;
            line-height: 1.6;
        }

        .result-main {
            display: flex;
            flex-direction: column;
            padding: 1rem;
            min-height: 198px;
            border-radius: 18px;
            background: var(--surface-soft);
            border: 1px solid var(--line);
        }

        .result-main-body {
            display: flex;
            flex-direction: column;
            gap: 0.9rem;
            flex: 1;
        }

        .result-main-primary {
            min-width: 0;
            flex: 1;
        }

        .result-main-label {
            color: var(--muted-soft);
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.42rem;
        }

        .result-main-class {
            color: var(--text);
            font-size: 1.42rem;
            font-weight: 600;
            line-height: 1.18;
            letter-spacing: -0.02em;
            min-height: 5rem;
            overflow: hidden;
            overflow-wrap: break-word;
            word-break: normal;
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
        }

        .result-main-side {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.5rem;
            min-width: 0;
            width: 100%;
        }

        .result-focus-stat {
            padding: 0.6rem 0.68rem;
            border-radius: 12px;
            background: rgba(201, 100, 66, 0.08);
            border: 1px solid rgba(201, 100, 66, 0.14);
        }

        .result-focus-stat.alt {
            background: rgba(20, 20, 19, 0.035);
            border-color: rgba(20, 20, 19, 0.06);
        }

        .result-focus-stat .label {
            color: var(--muted-soft);
            font-size: 0.68rem;
            margin-bottom: 0.18rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .result-focus-stat .value {
            color: var(--text);
            font-size: 0.95rem;
            font-weight: 600;
            line-height: 1.18;
        }

        .result-metric-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.48rem;
        }

        .result-metric {
            border-radius: 12px;
            background: rgba(20, 20, 19, 0.03);
            padding: 0.56rem 0.62rem;
            border: 1px solid var(--line);
        }

        .result-metric .label {
            color: var(--muted-soft);
            font-size: 0.68rem;
            margin-bottom: 0.18rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .result-metric .value {
            color: var(--text);
            font-size: 0.9rem;
            font-weight: 600;
            line-height: 1.2;
        }

        .result-detail-note {
            color: var(--muted-soft);
            font-size: 0.84rem;
            line-height: 1.5;
        }

        .prob-list {
            display: grid;
            gap: 0.55rem;
        }

        .prob-card {
            border-radius: 14px;
            border: 1px solid rgba(20, 20, 19, 0.06);
            background: rgba(255, 255, 255, 0.54);
            padding: 0.78rem 0.82rem 0.74rem 0.82rem;
        }

        .prob-head {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 0.9rem;
            color: var(--text);
            font-size: 0.92rem;
            margin-bottom: 0.42rem;
        }

        .prob-label {
            flex: 1;
            min-width: 0;
            line-height: 1.45;
            overflow: hidden;
            overflow-wrap: anywhere;
            word-break: break-word;
        }

        .prob-value {
            flex-shrink: 0;
            color: var(--muted);
            font-weight: 600;
        }

        .prob-track {
            width: 100%;
            height: 8px;
            border-radius: 999px;
            background: #efe9dd;
            overflow: hidden;
        }

        .prob-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--accent-deep), var(--accent));
        }

        @media (max-width: 1180px) {
            .result-main-side {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        @media (max-width: 760px) {
            .result-main-side,
            .result-metric-grid {
                grid-template-columns: minmax(0, 1fr);
            }
        }

        .compare-summary-shell {
            margin-top: 1.9rem;
            padding: 1.05rem 1.1rem 1.1rem 1.1rem;
            border-radius: 20px;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.52);
            box-shadow: var(--ring);
        }

        .compare-summary-title {
            color: var(--text);
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }

        .compare-summary-copy {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.6;
        }

        .compare-summary-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin-top: 0.9rem;
        }

        .compare-summary-card {
            padding: 0.82rem 0.82rem 0.78rem 0.82rem;
            border-radius: 14px;
            background: var(--surface-soft);
            border: 1px solid var(--line);
        }

        .compare-summary-card .label {
            color: var(--muted-soft);
            font-size: 0.74rem;
            margin-bottom: 0.28rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .compare-summary-card .value {
            color: var(--text);
            font-size: 1.04rem;
            font-weight: 600;
            line-height: 1.3;
        }

        .architecture {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.9rem;
            margin-top: 0.75rem;
        }

        .arch-node {
            position: relative;
            padding: 1.1rem 0.95rem 1rem 0.95rem;
            min-height: 148px;
            border-radius: 18px;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.7), rgba(250, 249, 245, 0.92));
            border: 1px solid rgba(20, 20, 19, 0.06);
            box-shadow: 0 10px 28px rgba(20, 20, 19, 0.04);
        }

        .arch-node::after {
            content: "→";
            position: absolute;
            right: -0.72rem;
            top: 50%;
            transform: translateY(-50%);
            color: rgba(94, 93, 89, 0.7);
            font-size: 1rem;
            font-weight: 700;
        }

        .architecture .arch-node:last-child::after {
            display: none;
        }

        .arch-step {
            color: var(--accent);
            font-size: 0.76rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.4rem;
        }

        .arch-title {
            color: var(--text);
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 0.35rem;
        }

        .arch-copy {
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.55;
        }

        .arch-detail-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.15fr) minmax(0, 0.85fr);
            gap: 1rem;
            align-items: stretch;
            margin: 2.85rem 0 1.65rem 0;
        }

        .arch-detail-card {
            display: flex;
            flex-direction: column;
            height: 100%;
            padding: 1.3rem 1.25rem 1.2rem 1.25rem;
            border-radius: 24px;
            background: rgba(255, 255, 255, 0.42);
            border: 1px solid rgba(20, 20, 19, 0.04);
        }

        .arch-detail-eyebrow {
            color: var(--muted-soft);
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.75rem;
        }

        .arch-detail-heading {
            color: var(--text);
            font-size: 1.1rem;
            font-weight: 600;
            margin: 0 0 0.45rem 0;
        }

        .arch-detail-copy {
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.72;
            margin: 0;
        }

        .arch-detail-points {
            display: grid;
            gap: 0.58rem;
            margin: 1rem 0 1.15rem 0;
        }

        .arch-detail-point {
            padding: 0.72rem 0.8rem;
            border-radius: 14px;
            background: rgba(20, 20, 19, 0.035);
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.55;
        }

        .arch-code-shell {
            margin-top: auto;
            padding: 0.88rem 0.95rem 0.95rem 0.95rem;
            border-radius: 18px;
            background: linear-gradient(180deg, #f7f4ed 0%, #f2eee5 100%);
            border: 1px solid rgba(20, 20, 19, 0.07);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.55);
        }

        .arch-code-caption {
            color: var(--muted-soft);
            font-size: 0.73rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.62rem;
        }

        .arch-code-shell pre {
            margin: 0;
            white-space: pre-wrap;
            color: #1f1f1d;
            font-size: 0.85rem;
            line-height: 1.7;
            font-family: "IBM Plex Mono", "JetBrains Mono", monospace;
        }

        .conclusion {
            padding: 0.9rem 0.95rem;
            border-radius: 16px;
            border: 1px solid var(--line);
            background: var(--surface);
            margin-bottom: 0.75rem;
            box-shadow: var(--ring);
        }

        .conclusion strong {
            color: var(--text);
        }

        .conclusion span {
            color: var(--muted);
            line-height: 1.65;
            font-size: 0.93rem;
        }

        .footer-note {
            color: var(--muted-soft);
            font-size: 0.88rem;
            line-height: 1.7;
            margin-top: 1.5rem;
            padding-top: 1rem;
            border-top: 1px solid var(--line);
        }

        @media (max-width: 1100px) {
            .compare-summary-grid,
            .architecture,
            .arch-detail-grid {
                grid-template-columns: 1fr;
            }

            .arch-node::after {
                display: none;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <section class="hero">
            <div class="small-chip">Primary target · Final student deployment</div>
            <div class="hero-title">TinyViT System Studio</div>
            <div class="hero-subtitle">
                面向轻量化图像分类毕业设计的推理工作台，聚焦最终部署 student、
                对比推理与论文主线结果复核。
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def stat_card(label: str, value: str, note: str) -> str:
    return f"""
    <div class="stat-card">
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}</div>
        <div class="stat-note">{note}</div>
    </div>
    """


def panel(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-title">{title}</div>
            <div class="panel-copy">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_probability_bars_html(items: list[dict[str, object]]) -> str:
    cards: list[str] = []
    for item in items:
        percent = float(item["prob"]) * 100.0
        label = escape(str(item["label"]))
        cards.append(
            normalize_html_fragment(
                f"""
            <div class="prob-card">
                <div class="prob-head">
                    <span class="prob-label">{label}</span>
                    <span class="prob-value">{percent:.2f}%</span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill" style="width: {percent:.2f}%"></div>
                </div>
            </div>
                """
            )
        )
    return normalize_html_fragment(
        f"""
        <div class="prob-list">
            {"".join(cards)}
        </div>
        """
    )


def build_result_card_html(result: dict[str, object], color: str) -> str:
    top1 = result["top_items"][0]
    return normalize_html_fragment(
        f"""
        <article class="result-shell">
            <div class="result-title">
                <div>
                    <div class="small-chip" style="margin-bottom: 0.28rem; color:{color}; border-color:{color}55; background:{color}12;">{escape(str(result['badge']))}</div>
                    <div class="result-name">{escape(str(result['title']))}</div>
                    <div class="result-note">{escape(str(result['note']))}</div>
                </div>
            </div>
            <section class="result-main">
                <div class="result-main-body">
                    <div class="result-main-primary">
                        <div class="result-main-label">Top-1 Prediction</div>
                        <div class="result-main-class">{escape(str(top1['label']))}</div>
                    </div>
                    <div class="result-main-side">
                        <div class="result-focus-stat">
                            <div class="label">Confidence</div>
                            <div class="value">{float(top1['prob']) * 100.0:.2f}%</div>
                        </div>
                        <div class="result-focus-stat alt">
                            <div class="label">Latency</div>
                            <div class="value">{float(result['elapsed_ms']):.2f} ms</div>
                        </div>
                    </div>
                </div>
            </section>
            <section class="result-metric-grid">
                <div class="result-metric"><div class="label">Val Top-1</div><div class="value">{float(result['best_val_acc']):.2f}%</div></div>
                <div class="result-metric"><div class="label">Params</div><div class="value">{float(result['params_m']):.2f} M</div></div>
                <div class="result-metric"><div class="label">FLOPs</div><div class="value">{float(result['flops_g']):.2f} G</div></div>
            </section>
        </article>
        """
    )


def render_result_panel(result: dict[str, object], color: str) -> None:
    st.markdown(build_result_card_html(result, color), unsafe_allow_html=True)
    detail_items = list(result["top_items"][1:])
    detail_count = len(detail_items)
    with st.expander(f"查看 {escape(str(result['title']))} 的 Top-K 详细概率", expanded=False):
        if detail_items:
            st.markdown(
                f'<div class="result-detail-note">Top-1 已常驻展示，这里仅展开其余 {detail_count} 个候选类别。</div>',
                unsafe_allow_html=True,
            )
            st.markdown(build_probability_bars_html(detail_items), unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="result-detail-note">当前返回结果只包含 Top-1，暂无更多候选类别可展开。</div>',
                unsafe_allow_html=True,
            )


def build_compare_summary_html(primary: dict[str, object], secondary: dict[str, object]) -> str:
    same_top1 = primary["top_items"][0]["wnid"] == secondary["top_items"][0]["wnid"]
    verdict = "一致" if same_top1 else "不同"
    latency_delta = float(primary["elapsed_ms"]) - float(secondary["elapsed_ms"])
    flops_delta = float(primary["flops_g"]) - float(secondary["flops_g"])
    return normalize_html_fragment(
        f"""
        <section class="compare-summary-shell">
            <div class="compare-summary-title">对比摘要</div>
            <div class="compare-summary-copy">同一输入图片下，对比主模型与参考模型的预测一致性、推理时延与复杂度差异。</div>
            <div class="compare-summary-grid">
                <div class="compare-summary-card"><div class="label">Top-1 一致性</div><div class="value">{verdict}</div></div>
                <div class="compare-summary-card"><div class="label">延迟差</div><div class="value">{latency_delta:+.2f} ms</div></div>
                <div class="compare-summary-card"><div class="label">复杂度差</div><div class="value">{flops_delta:+.2f} G</div></div>
            </div>
        </section>
        """
    )


def build_accuracy_flops_figure() -> plt.Figure:
    df = build_scatter_frame()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.set_facecolor("#faf9f5")
    fig.patch.set_facecolor("#faf9f5")
    for _, row in df.iterrows():
        ax.scatter(row["flops"], row["top1"], s=120, color=row["color"], edgecolors="#faf9f5", linewidths=1.0)
        ax.text(
            row["flops"] + 0.015,
            row["top1"] + 0.05,
            row["name"],
            fontsize=9.5,
            color="#3d3d3a",
        )
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35, color="#d1cfc5")
    ax.set_xlabel("FLOPs (G)", color="#3d3d3a")
    ax.set_ylabel("Validation Top-1 (%)", color="#3d3d3a")
    ax.set_title("Accuracy vs FLOPs", color="#141413", fontsize=14)
    ax.tick_params(colors="#3d3d3a")
    for spine in ax.spines.values():
        spine.set_color("#d1cfc5")
    return fig


def get_overview_metrics() -> dict[str, float | None]:
    student = merge_overview_metadata("student_final")
    teacher = merge_overview_metadata("teacher_final")
    baseline = merge_overview_metadata("baseline_112")
    return {
        "student_top1": safe_float(student.get("best_val_acc")),
        "student_params": safe_float(student.get("params_m")),
        "student_flops": safe_float(student.get("flops_g")),
        "teacher_top1": safe_float(teacher.get("best_val_acc")),
        "baseline_top1": safe_float(baseline.get("best_val_acc")),
        "baseline_flops": safe_float(baseline.get("flops_g")),
    }


def render_overview() -> None:
    metrics = get_overview_metrics()
    flop_reduction = compute_flop_reduction(metrics["student_flops"], metrics["baseline_flops"])
    teacher_gap = compute_teacher_gap(metrics["student_top1"], metrics["teacher_top1"])

    cols = st.columns(4)
    cards = [
        stat_card("Deployed student", format_overview_metric(metrics["student_top1"], suffix="%"), "最终学生模型验证精度。"),
        stat_card("Model size", format_overview_metric(metrics["student_params"], suffix="M"), "最终 student 参数量。"),
        stat_card("FLOPs reduction", format_overview_metric(flop_reduction, digits=1, suffix="%"), "相对 112 baseline 的推理复杂度下降。"),
        stat_card("Teacher gap", format_overview_metric(teacher_gap, suffix=" pp"), "最终 student 与最终 teacher 的精度差。"),
    ]
    for col, html in zip(cols, cards):
        with col:
            st.markdown(html, unsafe_allow_html=True)


def load_display_image(uploaded_file) -> Image.Image:
    if uploaded_file is not None:
        return Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
    if SAMPLE_IMAGE_PATH.exists():
        return Image.open(SAMPLE_IMAGE_PATH).convert("RGB")
    return Image.new("RGB", (112, 112), color=(19, 30, 24))


def render_live_inference(primary_key: str, compare_key: str | None, device: str, topk: int) -> None:
    st.markdown("### 实时推理工作台")
    left, right = st.columns([0.95, 1.05], gap="large")

    with left:
        upload = st.file_uploader(
            "上传单张图片进行推理",
            type=["jpg", "jpeg", "png", "bmp"],
            key="single_image_upload",
        )
        image = load_display_image(upload)
        st.image(image, caption="当前推理图片", use_container_width=True)
        with st.expander("ℹ️ 查看预处理协议"):
            st.markdown(
                "页面调用模型配置中的验证集预处理。当前主要部署目标为 112 输入主线，"
                "保持 resize、插值和归一化与实验评估一致，避免系统演示与论文结果脱节。"
            )

    with right:
        keys = [primary_key] + ([compare_key] if compare_key else [])
        keys = [key for key in keys if key]
        results: list[tuple[object, dict[str, object]]] = []
        for model_key in keys:
            spec = MODEL_SPECS[model_key]
            try:
                with st.spinner(f"加载 {spec.title} 并执行推理..."):
                    result = predict_image(model_key, image=image, device=device, topk=topk)
            except CheckpointLoadError as exc:
                st.error(f"{spec.title} 加载失败\n\n{exc}")
                continue
            results.append((spec, result))

        if results:
            result_cols = st.columns(len(results), gap="medium")
            for col, (spec, result) in zip(result_cols, results):
                with col:
                    render_result_panel(result, spec.color)

        if len(results) == 2:
            st.markdown(build_compare_summary_html(results[0][1], results[1][1]), unsafe_allow_html=True)


def render_batch_inference(device: str, topk: int) -> None:
    st.markdown("### 批量推理")
    model_options = {spec.title: key for key, spec in MODEL_SPECS.items()}
    selected_title = st.selectbox(
        "批量推理模型",
        list(model_options.keys()),
        index=list(model_options.values()).index("student_final"),
        key="batch_model_select",
    )
    selected_key = model_options[selected_title]
    uploads = st.file_uploader(
        "批量上传图片",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        key="batch_image_upload",
    )

    if uploads:
        try:
            with st.spinner(f"使用 {selected_title} 进行批量推理..."):
                df = predict_batch(selected_key, files=uploads, device=device, topk=topk)
        except CheckpointLoadError as exc:
            st.error(f"{selected_title} 加载失败\n\n{exc}")
            return
        c1, c2, c3 = st.columns(3)
        c1.metric("样本数量", f"{len(df)}")
        c2.metric("平均置信度", f"{df['top1_prob'].mean() * 100.0:.2f}%")
        c3.metric("平均延迟", f"{df['latency_ms'].mean():.2f} ms")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "下载批量推理结果 CSV",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name="tinyvit_batch_results.csv",
            mime="text/csv",
        )
    else:
        panel(
            "批量模式说明",
            "该模块用于论文系统测试场景。上传多张图片后，系统统一执行预处理、模型推理和 Top-5 汇总，"
            "可直接导出 CSV 作为附录结果或系统测试记录。",
        )


def render_conclusions() -> None:
    conclusions = [
        ("112 主线成立", "112 baseline 在更低复杂度下已经明显优于 224 标准 baseline，因此系统主入口默认围绕 112 主线展开。"),
        ("最终部署模型明确", "depth10 + hard distill + two-stage refine 的最终 student 以 4.60M / 1.77G 保持 79.41% Top-1，是最合适的系统部署对象。"),
        ("结构改进已收敛", "overlap patch12 是 112 主线中唯一稳定带来增益的结构改进，后续系统展示以其 teacher/student 结果为中心。"),
        ("train loss 不作为主图", "当前主线使用 mixup、cutmix、label smoothing 和 EMA，因此 val acc 曲线比 train loss 更适合作为论文主体可视化。"),
    ]
    for title, body in conclusions:
        st.markdown(
            f"""
            <div class="conclusion">
                <strong>{title}</strong><br/>
                <span>{body}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_experiment_dashboard() -> None:
    st.markdown("### 实验结果看板")
    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        fig = build_accuracy_flops_figure()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        render_conclusions()

    visual_dir = latest_thesis_visual_dir()
    if visual_dir is None:
        panel("可视化资源缺失", "尚未找到 `results/figures/*/thesis_visuals` 目录。请先运行 `python tools/generate_thesis_visuals.py`。")
        return

    st.markdown("#### 论文曲线与表格")
    curve_files = [
        visual_dir / "curve_mainline_overview.png",
        visual_dir / "curve_student_selection.png",
        visual_dir / "curve_structure_ablation_112.png",
        visual_dir / "curve_teacher_evolution.png",
    ]
    curve_captions = [
        "主线模型收敛曲线",
        "最终 student 选型曲线",
        "112 主线结构消融曲线",
        "teacher 演化曲线",
    ]
    cols = st.columns(2, gap="large")
    for idx, image_path in enumerate(curve_files):
        with cols[idx % 2]:
            if image_path.exists():
                st.image(str(image_path), caption=curve_captions[idx], use_container_width=True)

    table_main = load_csv_if_exists(visual_dir / "table_main_results.csv")
    table_student = load_csv_if_exists(visual_dir / "table_student_selection.csv")
    if table_main is not None and table_student is not None:
        t1, t2 = st.columns(2, gap="large")
        with t1:
            st.markdown("#### 主结果表")
            st.dataframe(table_main, use_container_width=True, hide_index=True)
        with t2:
            st.markdown("#### Student 选型表")
            st.dataframe(table_student, use_container_width=True, hide_index=True)

    panel(
        "关于 train loss",
        "系统页面刻意不将 train loss 作为论文主图，因为主线训练广泛使用 mixup、cutmix、label smoothing 和 EMA。"
        "这些操作会改变训练损失的量纲和解释方式，使它不再适合被直接拿来解释“是否过拟合”。因此系统主看板统一展示 val acc 曲线和精度-复杂度对比。",
    )


def render_architecture() -> None:
    structure_block = (
        "app/\n"
        "  streamlit_app.py      # 页面入口\n"
        "  system_runtime.py     # 模型加载、推理、结果整理\n"
        "configs/                # 模型配置\n"
        "models/                 # 模型定义\n"
        "results/                # checkpoint、summary、figures"
    )
    launch_command = "streamlit run app/streamlit_app.py"

    st.markdown("### 系统设计与实现")
    st.markdown(
        f"""
        <div class="small-chip">System Flow · Deployment-aligned inference path</div>
        <div class="architecture">
            <section class="arch-node">
                <div class="arch-step">Step 1</div>
                <div class="arch-title">Image Input</div>
                <div class="arch-copy">支持单图上传、批量上传和默认示例图，作为系统输入层。</div>
            </section>
            <section class="arch-node">
                <div class="arch-step">Step 2</div>
                <div class="arch-title">Preprocess</div>
                <div class="arch-copy">按模型配置复现验证集预处理，保持 resize、插值和归一化与实验一致。</div>
            </section>
            <section class="arch-node">
                <div class="arch-step">Step 3</div>
                <div class="arch-title">Model Router</div>
                <div class="arch-copy">根据用户选择调度 baseline、teacher 或 final student，并缓存权重。</div>
            </section>
            <section class="arch-node">
                <div class="arch-step">Step 4</div>
                <div class="arch-title">Inference Engine</div>
                <div class="arch-copy">执行 PyTorch 前向推理，输出 logits、Top-k 概率和单次时延。</div>
            </section>
            <section class="arch-node">
                <div class="arch-step">Step 5</div>
                <div class="arch-title">Result Board</div>
                <div class="arch-copy">展示类别、置信度、模型复杂度、实验结论和批量推理结果下载。</div>
            </section>
        </div>
        <div class="arch-detail-grid">
            <section class="arch-detail-card">
                <div class="arch-detail-eyebrow">Architecture Notes</div>
                <h4 class="arch-detail-heading">模块划分</h4>
                <p class="arch-detail-copy">
                    表现层由 Streamlit 构成；业务层负责预处理、模型调度、Top-k 后处理与结果组织；
                    模型层复用现有 <code>models/</code>、<code>configs/</code> 和训练得到的 checkpoint；
                    数据层使用 Tiny-ImageNet 类别映射、实验摘要和结果可视化文件。
                </p>
                <div class="arch-code-shell">
                    <div class="arch-code-caption">Project Structure</div>
                    <pre><code>{escape(structure_block)}</code></pre>
                </div>
            </section>
            <section class="arch-detail-card">
                <div class="arch-detail-eyebrow">Implementation Notes</div>
                <h4 class="arch-detail-heading">实现重点</h4>
                <p class="arch-detail-copy">
                    系统不重新实现训练主流程，而是直接复用主实验配置、权重与预处理协议，
                    让系统演示结果与论文实验结果保持同一套输入规范与评价口径。
                </p>
                <div class="arch-detail-points">
                    <div class="arch-detail-point">复用主实验配置，避免系统页面和训练脚本出现双份逻辑。</div>
                    <div class="arch-detail-point">推理结果、复杂度指标和实验看板统一指向论文主线资产。</div>
                    <div class="arch-detail-point">部署侧仅负责调度、展示与导出，不改变模型本身定义。</div>
                </div>
                <div class="arch-code-shell">
                    <div class="arch-code-caption">Launch Command</div>
                    <pre><code>{escape(launch_command)}</code></pre>
                </div>
            </section>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 系统功能清单")
    feature_cols = st.columns(4)
    features = [
        ("模型切换", "支持 224 baseline、112 baseline、final teacher、final student。"),
        ("单图推理", "支持单图上传与对比推理，输出 Top-5、耗时与复杂度。"),
        ("批量推理", "支持多图批量分析与 CSV 导出。"),
        ("结果看板", "集成论文曲线、表格与精度-复杂度关系图。"),
    ]
    for col, (title, body) in zip(feature_cols, features):
        with col:
            panel(title, body)
    with st.expander("查看模型目录"):
        catalog = build_model_catalog()
        st.dataframe(catalog, use_container_width=True, hide_index=True)


def render_footer() -> None:
    st.markdown(
        """
        <div class="footer-note">
            Design direction · Claude-inspired warm editorial layout from DESIGN.md<br/>
            Input protocol · Tiny-ImageNet 64 → 112 validation preprocessing<br/>
            System layer · Streamlit + PyTorch inference, aligned with thesis checkpoints and summaries
        </div>
        """,
        unsafe_allow_html=True,
    )


def resolve_default_compare_title(primary_title: str, secondary_options: list[str]) -> str | None:
    preferred_titles = [
        "Patch8 112 Baseline",
        "Student D10 Final",
        "Teacher Two-Stage",
        "DeiT 224 Baseline",
    ]
    for title in preferred_titles:
        if title != primary_title and title in secondary_options:
            return title
    return secondary_options[0] if secondary_options else None


def main() -> None:
    inject_theme()

    st.sidebar.markdown("## System Control")
    has_cuda = torch.cuda.is_available()

    model_titles = {spec.title: key for key, spec in MODEL_SPECS.items()}
    primary_title = st.sidebar.selectbox(
        "Primary model",
        list(model_titles.keys()),
        index=list(model_titles.values()).index("student_final"),
    )
    compare_mode = st.sidebar.checkbox("Enable compare mode", value=True)
    compare_key = None
    if compare_mode:
        secondary_options = [title for title in model_titles if title != primary_title]
        default_secondary_title = resolve_default_compare_title(primary_title, secondary_options)
        if default_secondary_title is not None:
            secondary_title = st.sidebar.selectbox(
                "Compare against",
                secondary_options,
                index=secondary_options.index(default_secondary_title),
            )
            compare_key = model_titles[secondary_title]

    if has_cuda:
        device = st.sidebar.selectbox("Inference device", ["cuda", "cpu"], index=0)
    else:
        device = "cpu"
        st.sidebar.text_input("Inference device", value=device, disabled=True)
        st.sidebar.caption("当前部署环境未检测到 CUDA。Streamlit Cloud 默认仅支持 CPU 推理。")
    topk = st.sidebar.slider("Top-K outputs", min_value=3, max_value=5, value=5)
    st.sidebar.caption("建议部署对象：Student D10 Final")

    render_hero()
    render_overview()

    tabs = st.tabs(["实时推理", "批量推理", "实验看板", "系统设计"])
    with tabs[0]:
        render_live_inference(model_titles[primary_title], compare_key, device, topk)
    with tabs[1]:
        render_batch_inference(device, topk)
    with tabs[2]:
        render_experiment_dashboard()
    with tabs[3]:
        render_architecture()

    render_footer()


if __name__ == "__main__":
    main()
