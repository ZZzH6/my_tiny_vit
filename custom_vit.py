import math
import torch
import torch.nn as nn
import timm
from timm.layers import trunc_normal_

# ==========================================
# 优化点4: 量化友好算子 (Hardswish / Hardsigmoid 代替 SiLU / Sigmoid)
# ==========================================
class CoordinateAttention(nn.Module):
    """
    量化友好的 Coordinate Attention 模块。
    使用大名鼎鼎的 MobileNetV3 配方，全部采用 Hard 激活函数，扫清落地部署时的 INT8 量化障碍。
    """
    def __init__(self, inp, oup, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        
        # 使用 Hardswish 替代 SiLU
        self.act = nn.Hardswish(inplace=True) 
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        
        # 使用 Hardsigmoid 替代原生的 Sigmoid
        self.hardsigmoid = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        identity = x
        
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # 空间双向门控
        a_h = self.hardsigmoid(self.conv_h(x_h))
        a_w = self.hardsigmoid(self.conv_w(x_w))

        out = identity * a_w * a_h
        return out


# ==========================================
# 新增模块：带局部偏置的 ConvFFN (Local FFN)
# ==========================================
class ConvFFN(nn.Module):
    """
    带有 3x3 深度可分离卷积的 FFN，替换原版纯 Linear 的 MLP。
    """
    # 核心修改：在这里的最后加上 **kwargs
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.Hardswish, drop=0., **kwargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        
        # 核心改进：引入 3x3 Depthwise 卷积进行局部空间交互
        self.dwconv = nn.Conv2d(
            hidden_features, hidden_features, 
            kernel_size=3, stride=1, padding=1, groups=hidden_features
        )
        
        # 默认搭配量化友好的 Hardswish
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, N, C = x.shape
        # 推导图像的宽和高 (对于正方形图像 H=W)
        H = int(math.sqrt(N))
        W = H 

        # 1. Linear 升维
        x = self.fc1(x)
        
        # 2. 从 Sequence (B, N, C) 转化为 Image (B, C, H, W) 进行卷积
        x_2d = x.transpose(1, 2).view(B, -1, H, W)
        x_2d = self.dwconv(x_2d)
        
        # 3. 再次展平回 Sequence (B, C, H, W) -> (B, N, C)
        x = x_2d.flatten(2).transpose(1, 2)
        
        # 4. 激活与降维
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ==========================================
# 毕设核心：极致轻量化且性能强悍的定制度 ViT
# ==========================================

class CustomLightViT(nn.Module):
    def __init__(self, num_classes=100, embed_dim=256, depth=8, num_heads=4, drop_rate=0.1, drop_path_rate=0.2):
        super(CustomLightViT, self).__init__()
        
        dim_stage1 = embed_dim // 2  # 例如 128
        dim_stage2 = embed_dim       # 例如 256
        
        # 1. Conv Stem
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, dim_stage1 // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim_stage1 // 2),
            nn.Hardswish(inplace=True),
            
            nn.Conv2d(dim_stage1 // 2, dim_stage1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim_stage1),
            nn.Hardswish(inplace=True)
        )
        
        self.ca_early = CoordinateAttention(inp=dim_stage1, oup=dim_stage1, reduction=16)

        # Stage 1 
        self.cpe_stage1 = nn.Conv2d(dim_stage1, dim_stage1, kernel_size=3, stride=1, padding=1, groups=dim_stage1, bias=True)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        depth_s1 = depth // 2
        depth_s2 = depth - depth_s1
        
        # 替换为 ConvFFN
        self.blocks_stage1 = nn.Sequential(*[
            timm.models.vision_transformer.Block(
                dim=dim_stage1, 
                num_heads=num_heads, 
                mlp_ratio=3.0,  
                qkv_bias=True, 
                proj_drop=drop_rate, 
                attn_drop=0.0,  
                drop_path=dpr[i],
                mlp_layer=ConvFFN,           # <--- 注入自定义的 ConvFFN
                act_layer=nn.Hardswish       # <--- 统一采用量化友好激活函数
            )
            for i in range(depth_s1)
        ])
        
        # Patch Merging
        self.downsample = nn.Sequential(
            nn.BatchNorm2d(dim_stage1),
            nn.Conv2d(dim_stage1, dim_stage2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim_stage2),
            nn.Hardswish(inplace=True)
        )
        
        self.ca_mid = CoordinateAttention(inp=dim_stage2, oup=dim_stage2, reduction=8)

        # Stage 2
        self.cpe_stage2 = nn.Conv2d(dim_stage2, dim_stage2, kernel_size=3, stride=1, padding=1, groups=dim_stage2, bias=True)
        
        # 替换为 ConvFFN
        self.blocks_stage2 = nn.Sequential(*[
            timm.models.vision_transformer.Block(
                dim=dim_stage2, 
                num_heads=num_heads, 
                mlp_ratio=3.0, 
                qkv_bias=True, 
                proj_drop=drop_rate, 
                attn_drop=0.0, 
                drop_path=dpr[depth_s1 + i],
                mlp_layer=ConvFFN,           # <--- 注入自定义的 ConvFFN
                act_layer=nn.Hardswish       # <--- 统一采用量化友好激活函数
            )
            for i in range(depth_s2)
        ])
        
        self.norm = nn.LayerNorm(dim_stage2)
        self.ca_out = CoordinateAttention(inp=dim_stage2, oup=dim_stage2, reduction=8)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(dim_stage2, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.ca_early(x)
        
        x = x + self.cpe_stage1(x)
        B, C1, H1, W1 = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.blocks_stage1(x)
        
        x = x.transpose(1, 2).view(B, C1, H1, W1)
        x = self.downsample(x)
        x = self.ca_mid(x)
        
        x = x + self.cpe_stage2(x)
        B, C2, H2, W2 = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.blocks_stage2(x)
        x = self.norm(x)
        
        return x, H2, W2

    def forward(self, x):
        x, H, W = self.forward_features(x)
        B, N, C = x.shape
        
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.ca_out(x)
        x = self.global_pool(x).flatten(1)
        x = self.head_drop(x)
        x = self.head(x)
        return x

if __name__ == '__main__':
    model = CustomLightViT(num_classes=100)
    dummy_input = torch.randn(2, 3, 32, 32)
    output = model(dummy_input)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] CustomLightViT (Local FFN) 模型总参数量: {total_params / 1e6:.2f} M")
    print(f"[*] 前向传播测试成功，输出维度: {output.shape}")