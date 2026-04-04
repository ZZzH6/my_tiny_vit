from __future__ import annotations

import timm


def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool,
    drop_path_rate: float = 0.1,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
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
