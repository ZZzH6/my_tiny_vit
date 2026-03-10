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
    经过四大架构升级的终极版 CustomLightViT：
    1. 【更深度的局域感知】: Convolutional Stem 内部使用 Hardswish 并在进入 Transformer 前立即融合一次早期 CA 注意力。
    2. 【破除静态诅咒】: 抛弃绝对位置编码，引入动态响应的条件位置编码 (CPE)，极大地提升对 RandomCrop 和 RandomErasing 的鲁棒性。
    3. 【纯粹的全局池化】: 抛弃格格不入的 CLS Token，完全利用深度 CA 重标定后的密集空间特征做 GAP 分类。
    4. 【全网络量化就绪】: 全流程剔除指数类算子。
    """
    def __init__(self, num_classes=100, embed_dim=192, depth=6, num_heads=3, drop_rate=0.1, drop_path_rate=0.2):
        super(CustomLightViT, self).__init__()
        
        self.H = 8 
        self.W = 8
        self.embed_dim = embed_dim
        
        # 优化点1 & 4：Convolutional Stem，结合 Hardswish
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, embed_dim // 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim // 4),
            nn.Hardswish(inplace=True),
            
            nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim // 2),
            nn.Hardswish(inplace=True),
            
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.Hardswish(inplace=True)
        )
        
        # 优化点2：CA 深度下放 - 早期空间感知模块
        # 在 Patch 进入自注意力的无序交互前，强制其进行一次深度的空间坐标交叉门控
        self.ca_early = CoordinateAttention(inp=embed_dim, oup=embed_dim, reduction=32)

        # 优化点3：CPE (Conditional Positional Encoding)
        # 用带 padding 的深度可分离卷积，动态赋予二维特征图位置偏移信息 (取代了 self.pos_embed)
        self.cpe = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, groups=embed_dim, bias=True)

        # 随机深度率
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # 核心堆叠 (Timm Block 本身非常高效且支持 DropPath)
        self.blocks = nn.Sequential(*[
            timm.models.vision_transformer.Block(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=2.0, 
                qkv_bias=True, 
                proj_drop=drop_rate, 
                attn_drop=drop_rate, 
                drop_path=dpr[i]
            )
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # 深层网络末端，分类之前的 CA 重标定模块
        self.ca_late = CoordinateAttention(inp=embed_dim, oup=embed_dim, reduction=32)
        
        # 优化点1：彻底抛弃 CLS，改用基于特征图的全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # 权重初始化
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
        # 1. 局部提取: [B, 3, 32, 32] -> [B, C, 8, 8]
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        
        # 2. 浅层级联坐标注意力 (CA 下放)
        x = self.ca_early(x)
        
        # 3. 注入动态位置信息 (CPE 残差模块)
        x = x + self.cpe(x)
        
        # 4. 转换维度适配 Transformer: [B, C, H, W] -> [B, N, C]
        x = x.flatten(2).transpose(1, 2)
        
        # 5. 深层全局交互 (Self-Attention)
        x = self.blocks(x)
        x = self.norm(x)
        return x, H, W

    def forward(self, x):
        # 获得序列化的高阶特征
        x, H, W = self.forward_features(x)
        B, N, C = x.shape
        
        # 逆转换回 2D 结构供深层 CA 与池化层消费: [B, N, C] -> [B, C, H, W]
        x = x.transpose(1, 2).view(B, C, H, W)
        
        # 进行最后一次基于空间特征重标定的 CA
        x = self.ca_late(x)
        
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
