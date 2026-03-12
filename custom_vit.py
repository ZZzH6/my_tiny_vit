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
# 毕设核心：极致轻量化且性能强悍的定制度 ViT
# ==========================================

class CustomLightViT(nn.Module):
    """
    终极版 CustomLightViT (冲击 90%+) - 引入层级化架构与多尺度特征
    1. 【高分辨率浅层 Stage 1】: 16x16 解析度，使用前期 Transformer 捕获细粒度特征。
    2. 【Patch Merging】: 空间降采样 (16x16 -> 8x8) 且通道翻倍，构建金字塔结构。
    3. 【深层全局交互 Stage 2】: 8x8 解析度，高通道数，负责全局语义。
    4. 【双重 CPE 注入】: 在每个 Stage 提供局部位置感知。
    """
    def __init__(self, num_classes=100, embed_dim=256, depth=8, num_heads=4, drop_rate=0.1, drop_path_rate=0.2):
        super(CustomLightViT, self).__init__()
        
        # 初始特征维度 (适配金字塔结构)
        dim_stage1 = embed_dim // 2  # 例如 96
        dim_stage2 = embed_dim       # 例如 192
        
        # 1. 更加轻量的 Conv Stem (输出保留较高分辨率: 原尺寸 64x64 -> 16x16)
        # stride=2 产生 32x32, 再次 stride=2 产生 16x16
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, dim_stage1 // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim_stage1 // 2),
            nn.Hardswish(inplace=True),
            
            nn.Conv2d(dim_stage1 // 2, dim_stage1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim_stage1),
            nn.Hardswish(inplace=True)
        )
        
        self.ca_early = CoordinateAttention(inp=dim_stage1, oup=dim_stage1, reduction=16)

        # Stage 1 (高分辨率、较少通道)
        self.cpe_stage1 = nn.Conv2d(dim_stage1, dim_stage1, kernel_size=3, stride=1, padding=1, groups=dim_stage1, bias=True)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        depth_s1 = depth // 2
        depth_s2 = depth - depth_s1
        
        self.blocks_stage1 = nn.Sequential(*[
            timm.models.vision_transformer.Block(
                dim=dim_stage1, 
                num_heads=num_heads, 
                mlp_ratio=4.0,  # 恢复到高容量 4.0
                qkv_bias=True, 
                proj_drop=drop_rate, 
                attn_drop=0.0,  # 移除 attn_drop
                drop_path=dpr[i]
            )
            for i in range(depth_s1)
        ])
        
        # Patch Merging (空间减半 16x16->8x8，通道数翻倍 dim_s1->dim_s2)
        self.downsample = nn.Sequential(
            nn.BatchNorm2d(dim_stage1),
            nn.Conv2d(dim_stage1, dim_stage2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim_stage2),
            nn.Hardswish(inplace=True)
        )
        
        self.ca_mid = CoordinateAttention(inp=dim_stage2, oup=dim_stage2, reduction=8)

        # Stage 2 (低分辨率、高通道)
        self.cpe_stage2 = nn.Conv2d(dim_stage2, dim_stage2, kernel_size=3, stride=1, padding=1, groups=dim_stage2, bias=True)
        
        self.blocks_stage2 = nn.Sequential(*[
            timm.models.vision_transformer.Block(
                dim=dim_stage2, 
                num_heads=num_heads, 
                mlp_ratio=4.0, # 恢复到 4.0 
                qkv_bias=True, 
                proj_drop=drop_rate, 
                attn_drop=0.0, 
                drop_path=dpr[depth_s1 + i]
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
        # 1. Stem (输入 64x64 -> 16x16)
        x = self.patch_embed(x)
        x = self.ca_early(x)
        
        # ———— Stage 1 (High Res: 16x16) ————
        x = x + self.cpe_stage1(x)
        B, C1, H1, W1 = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.blocks_stage1(x)
        
        # ———— Patch Merging (16x16 -> 8x8) ————
        x = x.transpose(1, 2).view(B, C1, H1, W1)
        x = self.downsample(x)
        x = self.ca_mid(x)
        
        # ———— Stage 2 (Low Res: 8x8) ————
        x = x + self.cpe_stage2(x)
        B, C2, H2, W2 = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.blocks_stage2(x)
        x = self.norm(x)
        
        return x, H2, W2

    def forward(self, x):
        # 获得序列化的高阶特征
        x, H, W = self.forward_features(x)
        B, N, C = x.shape
        
        # 逆转换回 2D 结构供深层 CA 与池化层消费: [B, N, C] -> [B, C, H, W]
        x = x.transpose(1, 2).view(B, C, H, W)
        
        # 进行最后一次基于空间特征重标定的 CA
        x = self.ca_out(x)
        
        # 进行全局空间特征池化 (不再依赖孤零零的 CLS Token)
        x = self.global_pool(x).flatten(1)
        
        # 分类
        x = self.head_drop(x)
        x = self.head(x)
        return x

if __name__ == '__main__':
    # 测试脚本：确保模型能正常跑通，并观察参数数量
    model = CustomLightViT(num_classes=100)
    dummy_input = torch.randn(2, 3, 32, 32)
    output = model(dummy_input)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] CustomLightViT 模型总参数量: {total_params / 1e6:.2f} M")
    print(f"[*] 前向传播测试成功，输出维度: {output.shape}") 
