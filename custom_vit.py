import math
from functools import partial

import timm
import torch
import torch.nn as nn
from timm.layers import trunc_normal_


def _make_divisible(value, divisor=32):
    """Round channel dimensions so each stage keeps a clean head dimension."""
    return int(math.ceil(value / divisor) * divisor)


class ConvBNAct(nn.Sequential):
    def __init__(self, in_chs, out_chs, kernel_size=3, stride=1, padding=1, groups=1, act_layer=nn.SiLU):
        super().__init__(
            nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_chs),
            act_layer(inplace=True),
        )


class CoordinateAttention(nn.Module):
    """
    Lightweight spatial-channel attention.
    Compared with SE, it keeps directional spatial information, which is
    useful on small-resolution datasets such as CIFAR.
    """
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mid_chs = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mid_chs, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_chs)
        self.act = nn.SiLU(inplace=True)
        self.conv_h = nn.Conv2d(mid_chs, oup, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mid_chs, oup, kernel_size=1, bias=False)

    def forward(self, x):
        identity = x
        _, _, h, w = x.shape

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return identity * a_h * a_w


class ConvFFN(nn.Module):
    """
    MLP with an inserted depthwise convolution.
    It preserves the efficiency of a Transformer FFN while injecting local bias.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, **kwargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features,
            bias=True,
        )
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        bsz, num_tokens, channels = x.shape
        side = int(math.sqrt(num_tokens))
        if side * side != num_tokens:
            raise ValueError(f"ConvFFN expects square token maps, got {num_tokens} tokens.")

        x = self.fc1(x)
        x_2d = x.transpose(1, 2).reshape(bsz, -1, side, side)
        x_2d = self.dwconv(x_2d)
        x = x_2d.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchMerging(nn.Module):
    """
    Downsample with a residual low-frequency path.
    This is gentler than a single stride-2 conv on tiny 32x32 inputs.
    """
    def __init__(self, in_chs, out_chs):
        super().__init__()
        self.pre_norm = nn.BatchNorm2d(in_chs)
        self.reduce = nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=2, padding=1, bias=False)
        self.skip = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_chs, out_chs, kernel_size=1, bias=False),
        )
        self.post_norm = nn.BatchNorm2d(out_chs)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.pre_norm(x)
        x = self.reduce(x) + self.skip(x)
        x = self.post_norm(x)
        return self.act(x)


class TransformerStage(nn.Module):
    """
    Stage wrapper that keeps features in 2D form and injects depthwise
    convolutional positional encoding before each Transformer block.
    """
    def __init__(self, dim, depth, num_heads, mlp_ratio, drop_rate, drop_path_rates):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.pos_embeds = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=True)
            for _ in range(depth)
        ])
        self.blocks = nn.ModuleList([
            timm.models.vision_transformer.Block(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_drop=drop_rate,
                attn_drop=0.0,
                init_values=1e-5,
                drop_path=drop_path_rates[idx],
                act_layer=nn.GELU,
                norm_layer=norm_layer,
                mlp_layer=ConvFFN,
            )
            for idx in range(depth)
        ])

    def forward(self, x):
        bsz, channels, height, width = x.shape
        for pos_embed, block in zip(self.pos_embeds, self.blocks):
            x = x + pos_embed(x)
            tokens = x.flatten(2).transpose(1, 2)
            tokens = block(tokens)
            x = tokens.transpose(1, 2).reshape(bsz, channels, height, width)
        return x


class AttentionPool(nn.Module):
    """
    Learnable token pooling, stronger than plain average pooling but still cheap.
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Linear(dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.score(self.norm(x)), dim=1)
        return torch.sum(x * weights, dim=1)


class CustomLightViT(nn.Module):
    """
    CIFAR-friendly lightweight hierarchical Transformer.

    Key changes relative to the old version:
    1. Preserve more spatial detail early: 32x32 -> 16x16 after the stem.
    2. Use three stages instead of two, so token reduction is more gradual.
    3. Inject convolutional positional encoding before every block.
    4. Replace final GAP-only head with attention pooling over tokens.
    """
    def __init__(self, num_classes=100, embed_dim=256, depth=8, num_heads=4, drop_rate=0.1, drop_path_rate=0.15):
        super().__init__()

        stage_dims = [
            _make_divisible(embed_dim * 3 / 8),
            _make_divisible(embed_dim * 5 / 8),
            _make_divisible(embed_dim),
        ]
        stage_heads = [max(1, dim // 32) for dim in stage_dims]

        if depth >= 6:
            stage_depths = [2, depth - 4, 2]
        else:
            stage_depths = [max(1, depth // 3), max(1, depth // 3), max(1, depth - 2 * max(1, depth // 3))]
            while sum(stage_depths) > depth:
                stage_depths[-1] -= 1
            while sum(stage_depths) < depth:
                stage_depths[1] += 1

        drop_path_values = [x.item() for x in torch.linspace(0, drop_path_rate, sum(stage_depths))]
        dpr_slices = []
        start = 0
        for d in stage_depths:
            dpr_slices.append(drop_path_values[start:start + d])
            start += d

        self.stem = nn.Sequential(
            ConvBNAct(3, stage_dims[0] // 2, kernel_size=3, stride=1, padding=1, act_layer=nn.SiLU),
            ConvBNAct(stage_dims[0] // 2, stage_dims[0], kernel_size=3, stride=2, padding=1, act_layer=nn.SiLU),
        )
        self.stem_attn = CoordinateAttention(stage_dims[0], stage_dims[0], reduction=16)

        self.stage1 = TransformerStage(
            dim=stage_dims[0],
            depth=stage_depths[0],
            num_heads=stage_heads[0],
            mlp_ratio=2.0,
            drop_rate=drop_rate,
            drop_path_rates=dpr_slices[0],
        )

        self.downsample1 = PatchMerging(stage_dims[0], stage_dims[1])
        self.stage2 = TransformerStage(
            dim=stage_dims[1],
            depth=stage_depths[1],
            num_heads=stage_heads[1],
            mlp_ratio=3.0,
            drop_rate=drop_rate,
            drop_path_rates=dpr_slices[1],
        )

        self.downsample2 = PatchMerging(stage_dims[1], stage_dims[2])
        self.stage3 = TransformerStage(
            dim=stage_dims[2],
            depth=stage_depths[2],
            num_heads=stage_heads[2],
            mlp_ratio=3.0,
            drop_rate=drop_rate,
            drop_path_rates=dpr_slices[2],
        )

        self.out_attn = CoordinateAttention(stage_dims[2], stage_dims[2], reduction=8)
        self.norm = nn.LayerNorm(stage_dims[2])
        self.pool = AttentionPool(stage_dims[2])
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(stage_dims[2], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stem_attn(x)

        x = self.stage1(x)
        x = self.downsample1(x)

        x = self.stage2(x)
        x = self.downsample2(x)

        x = self.stage3(x)
        x = self.out_attn(x)

        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = self.pool(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head_drop(x)
        x = self.head(x)
        return x


if __name__ == '__main__':
    model = CustomLightViT(num_classes=100)
    dummy_input = torch.randn(2, 3, 32, 32)
    output = model(dummy_input)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Optimized CustomLightViT 模型总参数量: {total_params / 1e6:.2f} M")
    print(f"[*] 前向传播测试成功，输出维度: {output.shape}")
