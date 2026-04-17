from __future__ import annotations

import torch
from torch import nn


class PrePatchLocalAdapter(nn.Module):
    """Lightweight image-space local adapter before patch embedding.

    The adapter preserves the RGB output shape so the pretrained patch embedding
    can still be reused unchanged after the image-space enhancement.
    """

    def __init__(
        self,
        in_chans: int = 3,
        hidden_channels: int = 24,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd for symmetric padding, got {kernel_size}")
        if int(hidden_channels) <= 0:
            raise ValueError(f"hidden_channels must be positive, got {hidden_channels}")

        padding = kernel_size // 2
        in_chans = int(in_chans)
        hidden_channels = int(hidden_channels)

        self.expand = nn.Conv2d(
            in_chans,
            hidden_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
        self.act1 = nn.GELU()
        self.dwconv = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=hidden_channels,
        )
        self.act2 = nn.GELU()
        self.project = nn.Conv2d(
            hidden_channels,
            in_chans,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        nn.init.kaiming_normal_(self.expand.weight, mode="fan_out", nonlinearity="relu")
        if self.expand.bias is not None:
            nn.init.zeros_(self.expand.bias)
        nn.init.dirac_(self.dwconv.weight, groups=hidden_channels)
        if self.dwconv.bias is not None:
            nn.init.zeros_(self.dwconv.bias)
        nn.init.normal_(self.project.weight, std=1e-5)
        if self.project.bias is not None:
            nn.init.zeros_(self.project.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"PrePatchLocalAdapter expects [B, C, H, W] images, got shape {tuple(x.shape)}")

        residual = x
        x = self.expand(x)
        x = self.act1(x)
        x = self.dwconv(x)
        x = self.act2(x)
        x = self.project(x)
        return residual + x


__all__ = ["PrePatchLocalAdapter"]
