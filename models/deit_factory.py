from __future__ import annotations

from types import MethodType
from typing import Any, Iterable

import timm
import torch
from timm.models.vision_transformer import checkpoint_seq
from torch import nn

from .local_ffn import LocalFFN
from .precnn_adapter import PreCNNLocalAdapter
from .prepatch_adapter import PrePatchLocalAdapter


COMMON_MODEL_KEYS = {
    "name",
    "num_classes",
    "pretrained",
    "distilled",
    "drop_path_rate",
    "drop_rate",
    "attn_drop_rate",
}

DEFAULT_LOCAL_FFN_BLOCKS = (8, 9, 10, 11)


def _to_2tuple(value: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"Expected a pair of values, got: {value}")
        return int(value[0]), int(value[1])
    scalar = int(value)
    return scalar, scalar


def _normalize_block_indices(block_indices: Iterable[int] | None) -> tuple[int, ...]:
    if block_indices is None:
        return DEFAULT_LOCAL_FFN_BLOCKS
    return tuple(dict.fromkeys(int(index) for index in block_indices))


def _resolve_patch_grid_size(
    img_size: int | tuple[int, int] | list[int],
    patch_size: int | tuple[int, int] | list[int],
) -> tuple[int, int]:
    img_h, img_w = _to_2tuple(img_size)
    patch_h, patch_w = _to_2tuple(patch_size)
    if img_h % patch_h != 0 or img_w % patch_w != 0:
        raise ValueError(
            f"img_size {img_size} must be divisible by patch_size {patch_size} to form a patch grid"
        )
    return img_h // patch_h, img_w // patch_w


def _apply_local_ffn_to_blocks(
    model: nn.Module,
    grid_size: tuple[int, int],
    block_indices: Iterable[int] | None = None,
    kernel_size: int = 3,
) -> nn.Module:
    indices = _normalize_block_indices(block_indices)
    num_blocks = len(model.blocks)
    for block_index in indices:
        if block_index < 0 or block_index >= num_blocks:
            raise ValueError(f"local_ffn block index {block_index} is out of range for {num_blocks} blocks")

    # Version 1 only swaps the FFN in the last 4 blocks. This keeps the change
    # local, limits added compute, and targets deeper tokens with stronger
    # semantics for a clean single-variable comparison against the baseline.
    for block_index in indices:
        block = model.blocks[block_index]
        block.mlp = LocalFFN(block.mlp, grid_size=grid_size, kernel_size=kernel_size)
    return model


def _forward_features_with_local_adapters(
    self: nn.Module,
    x: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if hasattr(self, "pre_patch_local_adapter"):
        x = self.pre_patch_local_adapter(x)
    x = self.patch_embed(x)
    x = self._pos_embed(x)
    x = self.patch_drop(x)
    x = self.norm_pre(x)
    if hasattr(self, "pre_cnn_local_adapter"):
        x = self.pre_cnn_local_adapter(x)

    if attn_mask is not None:
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)
    elif self.grad_checkpointing and not torch.jit.is_scripting():
        x = checkpoint_seq(self.blocks, x)
    else:
        x = self.blocks(x)

    x = self.norm(x)
    return x


def _attach_pre_cnn_local_adapter(
    model: nn.Module,
    grid_size: tuple[int, int],
    kernel_size: int = 3,
) -> nn.Module:
    embed_dim = int(model.embed_dim)
    model.pre_cnn_local_adapter = PreCNNLocalAdapter(
        embed_dim=embed_dim,
        grid_size=grid_size,
        kernel_size=kernel_size,
    )
    model.forward_features = MethodType(_forward_features_with_local_adapters, model)
    return model


def _attach_pre_patch_local_adapter(
    model: nn.Module,
    in_chans: int = 3,
    hidden_channels: int = 24,
    kernel_size: int = 3,
    output_size: tuple[int, int] | None = None,
    interpolation_mode: str = "bicubic",
    upsample_position: str = "before",
) -> nn.Module:
    if hasattr(model.patch_embed, "proj") and hasattr(model.patch_embed.proj, "in_channels"):
        in_chans = int(model.patch_embed.proj.in_channels)
    model.pre_patch_local_adapter = PrePatchLocalAdapter(
        in_chans=int(in_chans),
        hidden_channels=int(hidden_channels),
        kernel_size=kernel_size,
        output_size=output_size,
        interpolation_mode=interpolation_mode,
        upsample_position=str(upsample_position),
    )
    model.forward_features = MethodType(_forward_features_with_local_adapters, model)
    return model


def _copy_matching_tensors(
    target_state: dict[str, torch.Tensor],
    source_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    updated_state = dict(target_state)
    for key, value in source_state.items():
        if key in updated_state and updated_state[key].shape == value.shape:
            updated_state[key] = value
    return updated_state


def _initialize_distilled_model_from_base(
    model: nn.Module,
    base_timm_name: str,
    base_model_kwargs: dict[str, Any],
) -> nn.Module:
    base_model = timm.create_model(
        base_timm_name,
        pretrained=True,
        **base_model_kwargs,
    )

    model_state = _copy_matching_tensors(model.state_dict(), base_model.state_dict())
    base_state = base_model.state_dict()

    if "cls_token" in base_state and "dist_token" in model_state:
        if model_state["dist_token"].shape == base_state["cls_token"].shape:
            model_state["dist_token"] = base_state["cls_token"].clone()

    if "pos_embed" in base_state and "pos_embed" in model_state:
        src_pos = base_state["pos_embed"]
        dst_pos = model_state["pos_embed"].clone()
        if (
            dst_pos.ndim == 3
            and src_pos.ndim == 3
            and dst_pos.shape[0] == src_pos.shape[0]
            and dst_pos.shape[2] == src_pos.shape[2]
            and dst_pos.shape[1] == src_pos.shape[1] + 1
        ):
            dst_pos[:, :1, :] = src_pos[:, :1, :]
            dst_pos[:, 1:2, :] = src_pos[:, :1, :]
            dst_pos[:, 2:, :] = src_pos[:, 1:, :]
            model_state["pos_embed"] = dst_pos

    if "head.weight" in base_state and "head_dist.weight" in model_state:
        if model_state["head_dist.weight"].shape == base_state["head.weight"].shape:
            model_state["head_dist.weight"] = base_state["head.weight"].clone()
    if "head.bias" in base_state and "head_dist.bias" in model_state:
        if model_state["head_dist.bias"].shape == base_state["head.bias"].shape:
            model_state["head_dist.bias"] = base_state["head.bias"].clone()

    model.load_state_dict(model_state)
    model.pretrained_init_source = "base_deit_pretrained"
    return model


def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool,
    distilled: bool = False,
    drop_path_rate: float = 0.0,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    img_size: int | None = None,
    patch_size: int | None = None,
    local_ffn: bool = False,
    local_ffn_blocks: Iterable[int] | None = None,
    local_ffn_kernel_size: int = 3,
    pre_cnn_local: bool = False,
    pre_cnn_kernel_size: int = 3,
    pre_patch_local: bool = False,
    pre_patch_hidden_channels: int = 24,
    pre_patch_kernel_size: int = 3,
    pre_patch_internal_upsample: bool = False,
    pre_patch_interp_mode: str = "bicubic",
    pre_patch_upsample_position: str = "before",
    **timm_extra_kwargs: Any,
):
    distilled = bool(distilled)
    if model_name == "deit_tiny":
        timm_name = "deit_tiny_distilled_patch16_224" if distilled else "deit_tiny_patch16_224"
        base_timm_name = "deit_tiny_patch16_224"
    elif model_name == "deit_tiny_localffn":
        timm_name = "deit_tiny_distilled_patch16_224" if distilled else "deit_tiny_patch16_224"
        base_timm_name = "deit_tiny_patch16_224"
        img_size = 224 if img_size is None else int(img_size)
        patch_size = 16 if patch_size is None else int(patch_size)
    elif model_name == "deit_tiny_patch4_64":
        timm_name = "deit_tiny_distilled_patch16_224" if distilled else "deit_tiny_patch16_224"
        base_timm_name = "deit_tiny_patch16_224"
        img_size = 64 if img_size is None else int(img_size)
        patch_size = 4 if patch_size is None else int(patch_size)
    elif model_name == "deit_tiny_patch4_64_localffn":
        timm_name = "deit_tiny_distilled_patch16_224" if distilled else "deit_tiny_patch16_224"
        base_timm_name = "deit_tiny_patch16_224"
        img_size = 64 if img_size is None else int(img_size)
        patch_size = 4 if patch_size is None else int(patch_size)
    elif model_name == "deit_tiny_patch4_64_precnn_localffn":
        timm_name = "deit_tiny_distilled_patch16_224" if distilled else "deit_tiny_patch16_224"
        base_timm_name = "deit_tiny_patch16_224"
        img_size = 64 if img_size is None else int(img_size)
        patch_size = 4 if patch_size is None else int(patch_size)
    elif model_name == "deit_tiny_patch8_64_precnn_localffn":
        timm_name = "deit_tiny_distilled_patch16_224" if distilled else "deit_tiny_patch16_224"
        base_timm_name = "deit_tiny_patch16_224"
        img_size = 64 if img_size is None else int(img_size)
        patch_size = 8 if patch_size is None else int(patch_size)
    elif model_name == "deit_tiny_patch8_64_prepatch_localffn":
        timm_name = "deit_tiny_distilled_patch16_224" if distilled else "deit_tiny_patch16_224"
        base_timm_name = "deit_tiny_patch16_224"
        img_size = 64 if img_size is None else int(img_size)
        patch_size = 8 if patch_size is None else int(patch_size)
    elif model_name == "deit_tiny_patch8_112":
        timm_name = "deit_tiny_distilled_patch16_224" if distilled else "deit_tiny_patch16_224"
        base_timm_name = "deit_tiny_patch16_224"
        img_size = 112 if img_size is None else int(img_size)
        patch_size = 8 if patch_size is None else int(patch_size)
    elif model_name == "deit_tiny_patch8_112_prepatch_localffn":
        timm_name = "deit_tiny_distilled_patch16_224" if distilled else "deit_tiny_patch16_224"
        base_timm_name = "deit_tiny_patch16_224"
        img_size = 112 if img_size is None else int(img_size)
        patch_size = 8 if patch_size is None else int(patch_size)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    model_kwargs = {
        "num_classes": num_classes,
        "drop_path_rate": drop_path_rate,
        "drop_rate": drop_rate,
        "attn_drop_rate": attn_drop_rate,
    }
    if model_name in {
        "deit_tiny_localffn",
        "deit_tiny_patch4_64",
        "deit_tiny_patch4_64_localffn",
        "deit_tiny_patch4_64_precnn_localffn",
        "deit_tiny_patch8_64_precnn_localffn",
        "deit_tiny_patch8_64_prepatch_localffn",
        "deit_tiny_patch8_112",
        "deit_tiny_patch8_112_prepatch_localffn",
    } and img_size is not None:
        model_kwargs["img_size"] = int(img_size)
    if model_name in {
        "deit_tiny_localffn",
        "deit_tiny_patch4_64",
        "deit_tiny_patch4_64_localffn",
        "deit_tiny_patch4_64_precnn_localffn",
        "deit_tiny_patch8_64_precnn_localffn",
        "deit_tiny_patch8_64_prepatch_localffn",
        "deit_tiny_patch8_112",
        "deit_tiny_patch8_112_prepatch_localffn",
    } and patch_size is not None:
        model_kwargs["patch_size"] = int(patch_size)
    # Allow student-model scaling knobs such as depth/embed_dim/num_heads/mlp_ratio
    # to flow through from config into timm.create_model without adding a custom
    # model implementation for each ablation.
    model_kwargs.update(timm_extra_kwargs)

    if distilled and pretrained:
        model = timm.create_model(
            timm_name,
            pretrained=False,
            **model_kwargs,
        )
        model = _initialize_distilled_model_from_base(
            model,
            base_timm_name=base_timm_name,
            base_model_kwargs=model_kwargs,
        )
    else:
        model = timm.create_model(
            timm_name,
            pretrained=pretrained,
            **model_kwargs,
        )
    enable_local_ffn = bool(local_ffn) or local_ffn_blocks is not None or model_name in {
        "deit_tiny_localffn",
        "deit_tiny_patch4_64_localffn",
        "deit_tiny_patch4_64_precnn_localffn",
        "deit_tiny_patch8_64_precnn_localffn",
        "deit_tiny_patch8_64_prepatch_localffn",
        "deit_tiny_patch8_112_prepatch_localffn",
    }
    enable_pre_cnn_local = bool(pre_cnn_local) or model_name in {
        "deit_tiny_patch4_64_precnn_localffn",
        "deit_tiny_patch8_64_precnn_localffn",
    }
    enable_pre_patch_local = bool(pre_patch_local) or model_name in {
        "deit_tiny_patch8_64_prepatch_localffn",
        "deit_tiny_patch8_112_prepatch_localffn",
    }

    if enable_local_ffn or enable_pre_cnn_local:
        grid_size = _resolve_patch_grid_size(
            img_size if img_size is not None else 64,
            patch_size if patch_size is not None else 4,
        )

    if enable_local_ffn:
        _apply_local_ffn_to_blocks(
            model,
            grid_size=grid_size,
            block_indices=local_ffn_blocks,
            kernel_size=int(local_ffn_kernel_size),
        )
    if enable_pre_cnn_local:
        _attach_pre_cnn_local_adapter(
            model,
            grid_size=grid_size,
            kernel_size=int(pre_cnn_kernel_size),
        )
    if enable_pre_patch_local:
        pre_patch_output_size = _to_2tuple(img_size) if bool(pre_patch_internal_upsample) and img_size is not None else None
        _attach_pre_patch_local_adapter(
            model,
            hidden_channels=int(pre_patch_hidden_channels),
            kernel_size=int(pre_patch_kernel_size),
            output_size=pre_patch_output_size,
            interpolation_mode=str(pre_patch_interp_mode),
            upsample_position=str(pre_patch_upsample_position),
        )
    return model


def build_model_from_cfg(model_cfg: dict[str, Any], pretrained_override: bool | None = None):
    pretrained = bool(model_cfg["pretrained"]) if pretrained_override is None else bool(pretrained_override)
    extra_model_kwargs = {key: value for key, value in model_cfg.items() if key not in COMMON_MODEL_KEYS}
    return build_model(
        model_name=str(model_cfg["name"]),
        num_classes=int(model_cfg["num_classes"]),
        pretrained=pretrained,
        distilled=bool(model_cfg.get("distilled", False)),
        drop_path_rate=float(model_cfg.get("drop_path_rate", 0.0)),
        drop_rate=float(model_cfg.get("drop_rate", 0.0)),
        attn_drop_rate=float(model_cfg.get("attn_drop_rate", 0.0)),
        **extra_model_kwargs,
    )


__all__ = [
    "COMMON_MODEL_KEYS",
    "DEFAULT_LOCAL_FFN_BLOCKS",
    "build_model",
    "build_model_from_cfg",
]
