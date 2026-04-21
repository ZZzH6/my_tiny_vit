from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PrePatchLocalAdapter(nn.Module):
    """Stronger image-space local adapter before patch embedding.

    The adapter keeps RGB input/output so the pretrained patch embedding can be
    reused unchanged. When a target spatial size is provided, the upsampling can
    happen either before or after the local branch.
    """

    def __init__(
        self,
        in_chans: int = 3,
        hidden_channels: int = 24,
        kernel_size: int = 3,
        output_size: tuple[int, int] | None = None,
        interpolation_mode: str = "bicubic",
        upsample_position: str = "before",
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd for symmetric padding, got {kernel_size}")
        if int(hidden_channels) <= 0:
            raise ValueError(f"hidden_channels must be positive, got {hidden_channels}")
        if output_size is not None and len(output_size) != 2:
            raise ValueError(f"output_size must be a pair when provided, got {output_size}")

        padding = kernel_size // 2
        in_chans = int(in_chans)
        hidden_channels = int(hidden_channels)
        self.output_size = None if output_size is None else (int(output_size[0]), int(output_size[1]))
        self.interpolation_mode = str(interpolation_mode).strip().lower()
        if self.interpolation_mode not in {"nearest", "bilinear", "bicubic"}:
            raise ValueError(
                f"Unsupported interpolation_mode for PrePatchLocalAdapter: {interpolation_mode}"
            )
        self.upsample_position = str(upsample_position).strip().lower()
        if self.upsample_position not in {"before", "after"}:
            raise ValueError(
                "upsample_position for PrePatchLocalAdapter must be 'before' or 'after', "
                f"got: {upsample_position}"
            )

        self.expand = nn.Conv2d(
            in_chans,
            hidden_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
        self.act1 = nn.GELU()
        self.dwconv1 = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=hidden_channels,
        )
        self.pwconv1 = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.act2 = nn.GELU()
        self.dwconv2 = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=hidden_channels,
        )
        self.pwconv2 = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.act3 = nn.GELU()
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
        nn.init.dirac_(self.dwconv1.weight, groups=hidden_channels)
        if self.dwconv1.bias is not None:
            nn.init.zeros_(self.dwconv1.bias)
        nn.init.zeros_(self.pwconv1.weight)
        if self.pwconv1.bias is not None:
            nn.init.zeros_(self.pwconv1.bias)
        nn.init.dirac_(self.dwconv2.weight, groups=hidden_channels)
        if self.dwconv2.bias is not None:
            nn.init.zeros_(self.dwconv2.bias)
        nn.init.zeros_(self.pwconv2.weight)
        if self.pwconv2.bias is not None:
            nn.init.zeros_(self.pwconv2.bias)
        nn.init.normal_(self.project.weight, std=1e-5)
        if self.project.bias is not None:
            nn.init.zeros_(self.project.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"PrePatchLocalAdapter expects [B, C, H, W] images, got shape {tuple(x.shape)}")

        need_resize = self.output_size is not None and tuple(x.shape[-2:]) != self.output_size
        align_corners = False if self.interpolation_mode in {"bilinear", "bicubic"} else None
        if need_resize and self.upsample_position == "before":
            x = F.interpolate(
                x,
                size=self.output_size,
                mode=self.interpolation_mode,
                align_corners=align_corners,
            )

        residual = x
        x = self.expand(x)
        x = self.act1(x)
        x = x + self.pwconv1(self.dwconv1(x))
        x = self.act2(x)
        x = x + self.pwconv2(self.dwconv2(x))
        x = self.act3(x)
        x = self.project(x)
        x = residual + x

        if need_resize and self.upsample_position == "after":
            x = F.interpolate(
                x,
                size=self.output_size,
                mode=self.interpolation_mode,
                align_corners=align_corners,
            )

        return x


__all__ = ["PrePatchLocalAdapter"]
