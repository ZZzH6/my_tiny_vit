from __future__ import annotations

from collections.abc import Sequence

import torch
from timm.layers import to_2tuple
from torch import nn

from models.deit_tiny_convstem import (
    ConvStemPatchEmbed,
    build_standard_deit_tiny,
    load_standard_deit_pretrained,
)


class LocalTokenMixer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        grid_size: tuple[int, int],
        kernel_size: int = 3,
        has_class_token: bool = True,
    ):
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"local mixer kernel_size must be a positive odd integer, got {kernel_size}")

        self.embed_dim = int(embed_dim)
        self.grid_size = to_2tuple(grid_size)
        self.has_class_token = bool(has_class_token)
        self.dwconv = nn.Conv2d(
            self.embed_dim,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=self.embed_dim,
            bias=True,
        )
        self.cls_proj = nn.Linear(self.embed_dim, self.embed_dim) if self.has_class_token else None
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.dwconv.weight, mode="fan_out", nonlinearity="relu")
        if self.dwconv.bias is not None:
            nn.init.zeros_(self.dwconv.bias)
        if self.cls_proj is not None:
            nn.init.trunc_normal_(self.cls_proj.weight, std=0.02)
            nn.init.zeros_(self.cls_proj.bias)

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        if self.has_class_token:
            cls_tokens, patch_tokens = x[:, :1], x[:, 1:]
        else:
            cls_tokens, patch_tokens = None, x

        batch_size, num_tokens, channels = patch_tokens.shape
        height, width = self.grid_size
        expected_tokens = height * width
        if num_tokens != expected_tokens:
            raise AssertionError(
                f"LocalTokenMixer expected {expected_tokens} patch tokens from grid {self.grid_size}, got {num_tokens}."
            )
        if channels != self.embed_dim:
            raise AssertionError(f"LocalTokenMixer expected embed_dim={self.embed_dim}, got {channels}.")

        patch_tokens = patch_tokens.transpose(1, 2).reshape(batch_size, channels, height, width)
        patch_tokens = self.dwconv(patch_tokens)
        patch_tokens = patch_tokens.flatten(2).transpose(1, 2)

        if cls_tokens is None:
            return patch_tokens

        cls_update = self.cls_proj(patch_tokens.mean(dim=1, keepdim=True))
        return torch.cat([cls_update, patch_tokens], dim=1)


def build_deit_tiny_convstem_localmixer(
    num_classes: int,
    pretrained: bool,
    drop_path_rate: float = 0.1,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    stem_channels: Sequence[int] = (24, 48, 96),
    local_mixer_blocks: int = 4,
    local_mixer_kernel_size: int = 3,
):
    model = build_standard_deit_tiny(
        num_classes=num_classes,
        pretrained=False,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
    )
    model.patch_embed = ConvStemPatchEmbed(
        img_size=model.patch_embed.img_size,
        patch_size=model.patch_embed.patch_size,
        in_chans=3,
        embed_dim=model.embed_dim,
        stem_channels=stem_channels,
        norm_layer=None,
    )

    local_mixer_blocks = int(local_mixer_blocks)
    if local_mixer_blocks < 0 or local_mixer_blocks > len(model.blocks):
        raise ValueError(
            f"local_mixer_blocks must be within [0, {len(model.blocks)}], got {local_mixer_blocks}."
        )

    has_class_token = getattr(model, "cls_token", None) is not None
    for block_index in range(local_mixer_blocks):
        model.blocks[block_index].attn = LocalTokenMixer(
            embed_dim=model.embed_dim,
            grid_size=model.patch_embed.grid_size,
            kernel_size=local_mixer_kernel_size,
            has_class_token=has_class_token,
        )

    if pretrained:
        load_standard_deit_pretrained(
            model,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            variant_name="deit_tiny_convstem_localmixer",
        )
    return model
