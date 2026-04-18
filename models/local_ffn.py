from __future__ import annotations

import torch
from torch import nn


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

        grid_h, grid_w = self.grid_size
        expected_tokens = grid_h * grid_w
        prefix_tokens = x.shape[1] - expected_tokens
        if prefix_tokens <= 0:
            raise ValueError(
                f"LocalFFN expects at least one prefix token plus {expected_tokens} image tokens, "
                f"got sequence length {x.shape[1]}"
            )

        prefix, img_tokens = x[:, :prefix_tokens, :], x[:, prefix_tokens:, :]
        if img_tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"LocalFFN expects {expected_tokens} image tokens for grid {self.grid_size}, "
                f"got {img_tokens.shape[1]}"
            )

        batch_size, _, hidden_dim = img_tokens.shape
        img_tokens = img_tokens.transpose(1, 2).reshape(batch_size, hidden_dim, grid_h, grid_w)
        img_tokens = self.dwconv(img_tokens)
        img_tokens = img_tokens.flatten(2).transpose(1, 2)

        # Prefix tokens (cls or cls+dist) have no 2D neighborhood on the patch
        # grid, so they bypass depthwise convolution and are concatenated back
        # after local mixing.
        x = torch.cat((prefix, img_tokens), dim=1)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


__all__ = ["LocalFFN"]
