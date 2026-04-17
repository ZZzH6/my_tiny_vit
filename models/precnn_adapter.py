from __future__ import annotations

import torch
from torch import nn


class PreCNNLocalAdapter(nn.Module):
    """Lightweight CNN adapter before the first Transformer block.

    The module only mixes patch tokens on the 2D patch grid. The cls token bypasses
    the convolutional path so the global aggregation route remains untouched.
    """

    def __init__(
        self,
        embed_dim: int,
        grid_size: tuple[int, int],
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd for symmetric padding, got {kernel_size}")

        self.grid_size = (int(grid_size[0]), int(grid_size[1]))
        self.norm = nn.LayerNorm(int(embed_dim))
        self.dwconv = nn.Conv2d(
            int(embed_dim),
            int(embed_dim),
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=int(embed_dim),
        )
        self.pwconv = nn.Conv2d(
            int(embed_dim),
            int(embed_dim),
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.act = nn.GELU()

        # Keep the added path near an identity at initialization so pretrained DeiT
        # weights remain usable and the adapter can be studied as a single variable.
        nn.init.dirac_(self.dwconv.weight, groups=int(embed_dim))
        if self.dwconv.bias is not None:
            nn.init.zeros_(self.dwconv.bias)
        nn.init.zeros_(self.pwconv.weight)
        if self.pwconv.bias is not None:
            nn.init.zeros_(self.pwconv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"PreCNNLocalAdapter expects [B, N, C] tokens, got shape {tuple(x.shape)}")

        cls_token, img_tokens = x[:, :1, :], x[:, 1:, :]
        grid_h, grid_w = self.grid_size
        expected_tokens = grid_h * grid_w
        if img_tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"PreCNNLocalAdapter expects {expected_tokens} image tokens for grid {self.grid_size}, "
                f"got {img_tokens.shape[1]}"
            )

        residual = img_tokens
        img_tokens = self.norm(img_tokens)

        batch_size, _, embed_dim = img_tokens.shape
        img_tokens = img_tokens.transpose(1, 2).reshape(batch_size, embed_dim, grid_h, grid_w)
        img_tokens = self.dwconv(img_tokens)
        img_tokens = self.pwconv(img_tokens)
        img_tokens = self.act(img_tokens)
        img_tokens = img_tokens.flatten(2).transpose(1, 2)
        img_tokens = residual + img_tokens

        return torch.cat((cls_token, img_tokens), dim=1)


__all__ = ["PreCNNLocalAdapter"]
