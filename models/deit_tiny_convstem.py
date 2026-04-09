from __future__ import annotations

from collections.abc import Sequence

import timm
import torch
from timm.layers import to_2tuple
from torch import nn


class ConvBnAct(nn.Sequential):
    def __init__(self, in_chs: int, out_chs: int, stride: int):
        super().__init__(
            nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_chs),
            nn.GELU(),
        )


class DepthwiseSeparableConvBnAct(nn.Sequential):
    def __init__(self, in_chs: int, out_chs: int, stride: int):
        super().__init__(
            nn.Conv2d(in_chs, in_chs, kernel_size=3, stride=stride, padding=1, groups=in_chs, bias=False),
            nn.BatchNorm2d(in_chs),
            nn.GELU(),
            nn.Conv2d(in_chs, out_chs, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_chs),
            nn.GELU(),
        )


class ConvStemPatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 192,
        stem_channels: Sequence[int] = (24, 48, 96),
        norm_layer=None,
    ):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        if self.patch_size != (16, 16):
            raise ValueError(
                f"ConvStemPatchEmbed currently expects patch_size=16 to match DeiT-Tiny, got {self.patch_size}."
            )

        stem_channels = tuple(int(channel) for channel in stem_channels)
        if len(stem_channels) != 3:
            raise ValueError(f"stem_channels must contain exactly 3 stages, got {stem_channels}")
        if any(channel <= 0 for channel in stem_channels):
            raise ValueError(f"stem_channels must be positive, got {stem_channels}")

        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = True
        self.strict_img_size = True
        self.dynamic_img_pad = False
        self.output_fmt = None

        self.proj = nn.Sequential(
            ConvBnAct(in_chans, stem_channels[0], stride=2),
            DepthwiseSeparableConvBnAct(stem_channels[0], stem_channels[1], stride=2),
            DepthwiseSeparableConvBnAct(stem_channels[1], stem_channels[2], stride=2),
            DepthwiseSeparableConvBnAct(stem_channels[2], embed_dim, stride=2),
        )
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        if (height, width) != self.img_size:
            raise AssertionError(
                f"Input size ({height}, {width}) doesn't match model image size {self.img_size}."
            )
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


def build_standard_deit_tiny(
    num_classes: int,
    pretrained: bool,
    drop_path_rate: float = 0.1,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
):
    return timm.create_model(
        "deit_tiny_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
    )


def load_standard_deit_pretrained(
    model: nn.Module,
    *,
    num_classes: int,
    drop_path_rate: float,
    drop_rate: float,
    attn_drop_rate: float,
    variant_name: str,
) -> None:
    source_model = build_standard_deit_tiny(
        num_classes=num_classes,
        pretrained=True,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
    )
    incompatible = model.load_state_dict(source_model.state_dict(), strict=False)
    del source_model

    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        print(
            f"[{variant_name}] 复用标准 DeiT-Tiny 预训练权重："
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
        if missing:
            print(f"[{variant_name}] missing 示例: {missing[:4]}")
        if unexpected:
            print(f"[{variant_name}] unexpected 示例: {unexpected[:4]}")


def build_deit_tiny_convstem(
    num_classes: int,
    pretrained: bool,
    drop_path_rate: float = 0.1,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    stem_channels: Sequence[int] = (24, 48, 96),
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
    if pretrained:
        load_standard_deit_pretrained(
            model,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            variant_name="deit_tiny_convstem",
        )
    return model
