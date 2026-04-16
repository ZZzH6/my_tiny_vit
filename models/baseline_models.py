from __future__ import annotations

from typing import Any, Iterable

import timm
import torch
from torch import nn


COMMON_MODEL_KEYS = {
    "name",
    "num_classes",
    "pretrained",
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


class LocalFFN(nn.Module):
    """Version-1 local FFN for DeiT.

    The baseline block structure stays intact and only the FFN branch is changed so
    the ablation isolates local token mixing without touching attention, patch
    embedding, or the classifier head.
    """

    def __init__(
        self,
        base_mlp: nn.Module,
        grid_size: tuple[int, int],
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd for symmetric padding, got {kernel_size}")

        self.grid_size = (int(grid_size[0]), int(grid_size[1]))
        self.fc1 = base_mlp.fc1
        self.act = base_mlp.act
        self.drop1 = base_mlp.drop1
        self.norm = base_mlp.norm
        self.fc2 = base_mlp.fc2
        self.drop2 = base_mlp.drop2

        hidden_dim = int(self.fc1.out_features)
        self.dwconv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=hidden_dim,
        )
        # Start from an identity-like local path so pretrained DeiT weights still
        # transfer cleanly and only the new depthwise kernel needs adaptation.
        nn.init.dirac_(self.dwconv.weight, groups=hidden_dim)
        if self.dwconv.bias is not None:
            nn.init.zeros_(self.dwconv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)

        cls_token, img_tokens = x[:, :1, :], x[:, 1:, :]
        grid_h, grid_w = self.grid_size
        expected_tokens = grid_h * grid_w
        if img_tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"LocalFFN expects {expected_tokens} image tokens for grid {self.grid_size}, "
                f"got {img_tokens.shape[1]}"
            )

        batch_size, _, hidden_dim = img_tokens.shape
        img_tokens = img_tokens.transpose(1, 2).reshape(batch_size, hidden_dim, grid_h, grid_w)
        img_tokens = self.dwconv(img_tokens)
        img_tokens = img_tokens.flatten(2).transpose(1, 2)

        # The cls token has no 2D neighborhood on the patch grid, so it bypasses
        # depthwise convolution and is concatenated back after local mixing.
        x = torch.cat((cls_token, img_tokens), dim=1)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


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


def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool,
    drop_path_rate: float = 0.0,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    img_size: int | None = None,
    patch_size: int | None = None,
    local_ffn_blocks: Iterable[int] | None = None,
    local_ffn_kernel_size: int = 3,
    **_: Any,
):
    if model_name == "deit_tiny":
        timm_name = "deit_tiny_patch16_224"
    elif model_name == "deit_tiny_localffn":
        timm_name = "deit_tiny_patch16_224"
        img_size = 224 if img_size is None else int(img_size)
        patch_size = 16 if patch_size is None else int(patch_size)
    elif model_name == "deit_tiny_patch4_64":
        timm_name = "deit_tiny_patch16_224"
        img_size = 64 if img_size is None else int(img_size)
        patch_size = 4 if patch_size is None else int(patch_size)
    elif model_name == "deit_tiny_patch4_64_localffn":
        timm_name = "deit_tiny_patch16_224"
        img_size = 64 if img_size is None else int(img_size)
        patch_size = 4 if patch_size is None else int(patch_size)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    model_kwargs = {
        "pretrained": pretrained,
        "num_classes": num_classes,
        "drop_path_rate": drop_path_rate,
        "drop_rate": drop_rate,
        "attn_drop_rate": attn_drop_rate,
    }
    if model_name in {"deit_tiny_localffn", "deit_tiny_patch4_64", "deit_tiny_patch4_64_localffn"} and img_size is not None:
        model_kwargs["img_size"] = int(img_size)
    if model_name in {"deit_tiny_localffn", "deit_tiny_patch4_64", "deit_tiny_patch4_64_localffn"} and patch_size is not None:
        model_kwargs["patch_size"] = int(patch_size)

    model = timm.create_model(
        timm_name,
        **model_kwargs,
    )
    if model_name in {"deit_tiny_localffn", "deit_tiny_patch4_64_localffn"}:
        img_h, img_w = _to_2tuple(img_size if img_size is not None else 64)
        patch_h, patch_w = _to_2tuple(patch_size if patch_size is not None else 4)
        grid_size = (img_h // patch_h, img_w // patch_w)
        _apply_local_ffn_to_blocks(
            model,
            grid_size=grid_size,
            block_indices=local_ffn_blocks,
            kernel_size=int(local_ffn_kernel_size),
        )
    return model


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
