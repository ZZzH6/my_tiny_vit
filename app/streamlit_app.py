from __future__ import annotations

import io
import json
from pathlib import Path

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

        .prob-card {
            border-radius: 16px;
            border: 1px solid var(--line);
            background: var(--surface);
            padding: 0.8rem 0.85rem 0.7rem 0.85rem;
            margin-bottom: 0.68rem;
            box-shadow: var(--ring);
        }

        .prob-head {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            color: var(--text);
            font-size: 0.94rem;
            margin-bottom: 0.45rem;
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

        .result-shell {
            border-radius: 18px;
            border: 1px solid var(--line);
            background: var(--surface);
            padding: 1rem;
            height: 100%;
            box-shadow: var(--ring), var(--shadow);
        }

        .result-title {
            display: flex;
            justify-content: space-between;
            gap: 0.8rem;
            align-items: center;
            margin-bottom: 0.7rem;
        }

        .result-name {
            color: var(--text);
            font-size: 1.1rem;
            font-weight: 600;
        }

        .result-note {
            color: var(--muted-soft);
            font-size: 0.9rem;
            margin-top: 0.25rem;
            line-height: 1.55;
        }

        .result-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.7rem;
            margin: 0.85rem 0 1rem 0;
        }

        .mini-stat {
            border-radius: 14px;
            background: var(--surface-soft);
            padding: 0.78rem 0.75rem;
            border: 1px solid var(--line);
        }

        .mini-stat .label {
            color: var(--muted-soft);
            font-size: 0.76rem;
            margin-bottom: 0.3rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .mini-stat .value {
            color: var(--text);
            font-size: 1.1rem;
            font-weight: 600;
        }

        .architecture {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.75rem;
            margin-top: 0.4rem;
        }

        .arch-node {
            position: relative;
            padding: 0.95rem 0.85rem;
            min-height: 130px;
            border-radius: 16px;
            background: var(--surface);
            border: 1px solid var(--line);
            box-shadow: var(--ring);
        }

        .arch-node::after {
            content: "→";
            position: absolute;
            right: -0.58rem;
            top: 50%;
            transform: translateY(-50%);
            color: var(--accent);
            font-size: 1.15rem;
            font-weight: 800;
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
            .result-grid,
            .architecture {
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


def render_probability_bars(items: list[dict[str, float]]) -> None:
    for item in items:
        percent = item["prob"] * 100.0
        st.markdown(
            f"""
            <div class="prob-card">
                <div class="prob-head">
                    <span>{item['label']}</span>
                    <span>{percent:.2f}%</span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill" style="width: {percent:.2f}%"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_result(result: dict[str, object], color: str) -> None:
    top1 = result["top_items"][0]
    st.markdown(
        f"""
        <div class="result-shell">
            <div class="result-title">
                <div>
                    <div class="small-chip" style="margin-bottom: 0.25rem; color:{color}; border-color:{color}55; background:{color}12;">{result['badge']}</div>
                    <div class="result-name">{result['title']}</div>
                    <div class="result-note">{result['note']}</div>
                </div>
            </div>
            <div class="result-grid">
                <div class="mini-stat"><div class="label">Top-1</div><div class="value">{top1['label']}</div></div>
                <div class="mini-stat"><div class="label">Confidence</div><div class="value">{top1['prob'] * 100.0:.2f}%</div></div>
                <div class="mini-stat"><div class="label">Latency</div><div class="value">{result['elapsed_ms']:.2f} ms</div></div>
                <div class="mini-stat"><div class="label">Val Top-1</div><div class="value">{result['best_val_acc']:.2f}%</div></div>
                <div class="mini-stat"><div class="label">Params</div><div class="value">{result['params_m']:.2f} M</div></div>
                <div class="mini-stat"><div class="label">FLOPs</div><div class="value">{result['flops_g']:.2f} G</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_probability_bars(result["top_items"])


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
    left, right = st.columns([0.88, 1.12], gap="large")

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
        result_cols = st.columns(len(keys), gap="large")
        results = []
        for idx, model_key in enumerate(keys):
            spec = MODEL_SPECS[model_key]
            with result_cols[idx]:
                try:
                    with st.spinner(f"加载 {spec.title} 并执行推理..."):
                        result = predict_image(model_key, image=image, device=device, topk=topk)
                except CheckpointLoadError as exc:
                    st.error(f"{spec.title} 加载失败\n\n{exc}")
                    continue
                results.append(result)
                render_result(result, spec.color)

        if len(results) == 2:
            primary = results[0]
            secondary = results[1]
            same_top1 = primary["top_items"][0]["wnid"] == secondary["top_items"][0]["wnid"]
            verdict = "一致" if same_top1 else "不同"
            st.markdown("#### 对比摘要")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Top-1 一致性", verdict)
            with c2:
                st.metric(
                    "延迟差",
                    f"{primary['elapsed_ms'] - secondary['elapsed_ms']:+.2f} ms",
                )
            with c3:
                st.metric(
                    "复杂度差",
                    f"{primary['flops_g'] - secondary['flops_g']:+.2f} G",
                )


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
    st.markdown("### 系统设计与实现")
    st.markdown(
        """
        <div class="architecture">
            <div class="arch-node">
                <div class="arch-step">Step 1</div>
                <div class="arch-title">Image Input</div>
                <div class="arch-copy">支持单图上传、批量上传和默认示例图，作为系统输入层。</div>
            </div>
            <div class="arch-node">
                <div class="arch-step">Step 2</div>
                <div class="arch-title">Preprocess</div>
                <div class="arch-copy">按模型配置复现验证集预处理，保持 resize、插值和归一化与实验一致。</div>
            </div>
            <div class="arch-node">
                <div class="arch-step">Step 3</div>
                <div class="arch-title">Model Router</div>
                <div class="arch-copy">根据用户选择调度 baseline、teacher 或 final student，并缓存权重。</div>
            </div>
            <div class="arch-node">
                <div class="arch-step">Step 4</div>
                <div class="arch-title">Inference Engine</div>
                <div class="arch-copy">执行 PyTorch 前向推理，输出 logits、Top-k 概率和单次时延。</div>
            </div>
            <div class="arch-node">
                <div class="arch-step">Step 5</div>
                <div class="arch-title">Result Board</div>
                <div class="arch-copy">展示类别、置信度、模型复杂度、实验结论和批量推理结果下载。</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        panel(
            "模块划分",
            "表现层由 Streamlit 构成；业务层负责预处理、模型调度、Top-k 后处理与结果组织；"
            "模型层复用现有 `models/`、`configs/` 和训练得到的 checkpoint；"
            "数据层使用 Tiny-ImageNet 类别映射、实验摘要和结果可视化文件。",
        )
        st.code(
            "app/\n"
            "  streamlit_app.py      # 页面入口\n"
            "  system_runtime.py     # 模型加载、推理、结果整理\n"
            "configs/                # 模型配置\n"
            "models/                 # 模型定义\n"
            "results/                # checkpoint、summary、figures",
            language="text",
        )
    with right:
        panel(
            "实现重点",
            "本系统没有重新实现训练逻辑，而是直接复用主实验配置、权重与预处理协议。"
            "这样可以保证系统演示结果与论文实验结果严格对齐，避免出现“实验一套、页面一套”的偏差。",
        )
        st.code("streamlit run app/streamlit_app.py", language="bash")

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


def main() -> None:
    inject_theme()

    st.sidebar.markdown("## System Control")
    device_candidates = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]

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
        secondary_title = st.sidebar.selectbox(
            "Compare against",
            secondary_options,
            index=secondary_options.index("Patch8 112 Baseline"),
        )
        compare_key = model_titles[secondary_title]

    device = st.sidebar.selectbox("Inference device", device_candidates, index=0)
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
