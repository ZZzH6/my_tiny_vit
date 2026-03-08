import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, PatchEmbed

# ==========================================
# 1. 局部注意力锚点: Coordinate Attention (CA)
# ==========================================
# 这个模块是你轻量化改造的“灵魂”。
# 原版的 ViT 纯靠自注意力，对局部特征捕捉很弱。我们在它内部插拔这个算子，弥补这部分缺陷。
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
    def __init__(self, num_classes=100, embed_dim=192, depth=6, num_heads=3, drop_rate=0.1):
        super(CustomLightViT, self).__init__()
        
        # 记录内部所需参数
        self.embed_dim = embed_dim
        
        # 1. Patch Embedding: 跟原版 ViT 一样，将图像切片 (16x16) 并投影
        # 输入: B x 3 x 224 x 224
        # 输出: B x L x embed_dim (即 B x 196 x 192)
        self.patch_embed = PatchEmbed(img_size=224, patch_size=16, in_chans=3, embed_dim=embed_dim)
        
        # ==【毕设魔改点1】：Patch 后的局部特征增强 ==
        # 一般 ViT 切完片就丢进 Transformer 了，这里我们利用 CA 进行一次空间位置标定！
        # 但 CA 需要 2D 图像特征 (B, C, H, W)，所以我们需要 Reshape。
        self.ca_module = CoordinateAttention(c_in=embed_dim, c_out=embed_dim)
        
        # 2. Class Token & Position Embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # 3. Transformer Encoder Blocks
        # 我们使用 timm 最底层的 VisionTransformer 骨架，但参数全部是我们极度砍伐过的
        self.blocks = VisionTransformer(
            patch_size=16, 
            embed_dim=embed_dim, 
            depth=depth, 
            num_heads=num_heads, 
            num_classes=0, # 我们自己做分类头
            global_pool='', 
            drop_rate=drop_rate, 
            attn_drop_rate=drop_rate
        ).blocks
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # ==【毕设魔改点2】：更强的自定义分类头 ==
        # 不再是一层 Linear，而是带 Dropout 保护的 MLP
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
        # 简单的权重初始化
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)

    def forward(self, x):
        B = x.shape[0]
        
        # 1. Patch Embed (变成 Sequence)
        x = self.patch_embed(x) # [B, 196, 192]
        
        # --- 毕设魔改区: 插入 CA 模块 ---
        # 必须还原回 2D 结构才能做卷积注意力。196 个 Patch = 14x14
        H = W = 14
        x_2d = x.transpose(1, 2).view(B, self.embed_dim, H, W) # [B, 192, 14, 14]
        x_2d = self.ca_module(x_2d) # 经过 Coordinate Attention
        x = x_2d.flatten(2).transpose(1, 2) # 又变回 [B, 196, 192]
        # -------------------------------
        
        # 2. 拼接 CLS Token 和位置编码
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # [B, 197, 192]
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 3. 逐层穿过削减版 Transformer
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # 4. 提取 CLS Token 作为整张图的全局特征
        cls_feature = x[:, 0] # [B, 192]
        
        # 5. 通过魔改版分类头输出预测
        out = self.head(cls_feature)
        
        return out


if __name__ == '__main__':
    # 快速测试脚本，确保我们在纸上设计的维度可以成功 Forward，且计算参数量
    model = CustomLightViT(num_classes=100)
    
    # 打印参数量信息以便对比
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[*] CustomLightViT 模型总参数量: {total_params / 1e6:.2f} M")
    
    dummy_input = torch.randn(2, 3, 224, 224)
    out = model(dummy_input)
    print(f"[*] 前向传播测试成功，输出维度: {out.shape}")
