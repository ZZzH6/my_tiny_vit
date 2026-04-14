from __future__ import annotations

from typing import Any

import timm


COMMON_MODEL_KEYS = {
    "name",
    "num_classes",
    "pretrained",
    "drop_path_rate",
    "drop_rate",
    "attn_drop_rate",
}


def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool,
    drop_path_rate: float = 0.0,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    **_: Any,
):
    if model_name != "deit_tiny":
        raise ValueError(f"Unsupported model_name: {model_name}")

    return timm.create_model(
        "deit_tiny_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
    )


def build_model_from_cfg(model_cfg: dict[str, Any], pretrained_override: bool | None = None):
    pretrained = bool(model_cfg["pretrained"]) if pretrained_override is None else bool(pretrained_override)
    extra_model_kwargs = {key: value for key, value in model_cfg.items() if key not in COMMON_MODEL_KEYS}
    return build_model(
        model_name=str(model_cfg["name"]),
        num_classes=int(model_cfg["num_classes"]),
        pretrained=pretrained,
        drop_path_rate=float(model_cfg.get("drop_path_rate", 0.0)),
        drop_rate=float(model_cfg.get("drop_rate", 0.0)),
        attn_drop_rate=float(model_cfg.get("attn_drop_rate", 0.0)),
        **extra_model_kwargs,
    )
