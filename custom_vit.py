import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, PatchEmbed
import timm

# ==========================================
# 1. 局部注意力锚点: Coordinate Attention (CA)
# ==========================================
class CoordinateAttention(nn.Module):
    """
    Coordinate Attention (坐标注意力) 模块
    作用：捕捉跨通道特征，并把空间坐标信息编码进特征图。
    适用性广，计算极小。
    """
    def __init__(self, c_in, c_out, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, c_in // reduction)

        self.conv1 = nn.Conv2d(c_in, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()
        
        self.conv_h = nn.Conv2d(mip, c_out, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, c_out, kernel_size=1, stride=1, padding=0)

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

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        out = identity * a_w * a_h
        return out


# ==========================================
# 2. 你的核心毕设模型：CustomLightViT (基于 ViT-Base 魔改)
# ==========================================
class CustomLightViT(nn.Module):
    """
    【毕设核心创新模型】 - 手工轻量化的 ViT
    思路来源：
    1. 抛弃庞大的参数，降低维度和深度。
       原版 ViT-Base: 12 层, dim=768, heads=12 (86M 参数)
       我们的定制版:   6 层, dim=192, heads=3  (估算 <5M 参数)
    2. 引入 Coordinate Attention 弥补 Transformer 处理短距离局部特征的弱点。
    """
    def __init__(self, num_classes=100, embed_dim=192, depth=6, num_heads=3, drop_rate=0.1, drop_path_rate=0.15):
        super(CustomLightViT, self).__init__()
        
        # 记录内部所需参数
        self.embed_dim = embed_dim
        
        # ==【毕设魔改点1】：将原生 PatchEmbed 替换为 Convolutional Stem ==
        # 原版 ViT 的 16x16 切片对细粒度/小图特征极度不友好。
        # 我们使用一个 3 层的轻量级卷积主干。由于数据集已改回原生的 32x32 尺寸：
        # 这里进行轻微下采样 (stride=(1,2,1))
        # 32x32 -> (stride=1) -> 32x32 -> (stride=2) -> 16x16 -> (stride=1) -> 16x16
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, embed_dim // 4, kernel_size=3, stride=1, padding=1, bias=False),  
            nn.BatchNorm2d(embed_dim // 4),
            nn.SiLU(inplace=True),
            
            nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(embed_dim // 2),
            nn.SiLU(inplace=True),
            
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1, bias=False), 
            nn.BatchNorm2d(embed_dim)
        )
        self.num_patches = 16 * 16  # 256
        self.H = 16
        self.W = 16
        
        # 2. Class Token & Position Embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # 3. Transformer Encoder Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            timm.models.vision_transformer.Block(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=4.0, 
                qkv_bias=True, 
                proj_drop=drop_rate,  # 修复 timm API 更新：drop => proj_drop
                attn_drop=drop_rate, 
                drop_path=dpr[i]
            )
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)

        # ==【毕设魔改点2】：深层特征融合 CA 模块 ==
        # 将 Coordinate Attention 从开头移到了 Transformer 提完特征之后！
        # 让它去重塑已经具备全局感受野的高阶语义特征，而不是原始像素框。
        self.ca_module = CoordinateAttention(c_in=embed_dim, c_out=embed_dim)
        
        # ==【毕设魔改点3】：更强的自定义分类头 ==
        hidden_dim = embed_dim // 2
        self.head = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)

    def forward(self, x):
        B = x.shape[0]
        
        # 1. Conv Stem (取代 Patch Embed)
        x = self.patch_embed(x) # [B, 192, 14, 14]
        
        # 将 2D Feature Map 展平为 Sequence: [B, 192, 196] -> [B, 196, 192]
        x = x.flatten(2).transpose(1, 2)
        
        # 2. 拼接 CLS Token 和位置编码
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # [B, 197, 192]
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 3. 逐层穿过削减版 Transformer
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # 4. 提取输出
        cls_feature = x[:, 0] # [B, 192]
        spatial_features = x[:, 1:] # [B, 256, 192]

        # --- 毕设魔改区: 在深层插入 CA 模块重塑空间特征 ---
        # 还原回 2D 结构 (16x16)
        spatial_features_2d = spatial_features.transpose(1, 2).view(B, self.embed_dim, self.H, self.W) # [B, 192, 16, 16]
        spatial_features_2d = self.ca_module(spatial_features_2d) # 深层 CA 加权
        
        # CA 加权后的空间特征进行池化 (Global Average Pooling)
        spatial_pooled = spatial_features_2d.flatten(2).mean(dim=2) # [B, 192]

        # 融合 CLS_Token 和 池化后的空间特征
        final_feature = cls_feature + spatial_pooled 
        # -------------------------------
        
        # 5. 通过魔改版分类头输出预测
        out = self.head(final_feature)
        
        return out


if __name__ == '__main__':
    # 快速测试脚本，确保我们在纸上设计的维度可以成功 Forward，且计算参数量
    model = CustomLightViT(num_classes=100)
    
    # 打印参数量信息以便对比
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[*] CustomLightViT 模型总参数量: {total_params / 1e6:.2f} M")
    
    dummy_input = torch.randn(2, 3, 32, 32)
    out = model(dummy_input)
    print(f"[*] 前向传播测试成功，输出维度: {out.shape}")

