import torch
import torch.nn as nn
import timm

# ==========================================
# 自定义即插即用的注意力模块 (论文创新点)
# ==========================================

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) 注意力机制模块
    作用：显式地建模通道之间的相互依赖关系，自适应地重新标定特征的通道权重。
    优势：参数量极小，即插即用，能有效提升轻量化模型对关键特征的捕捉能力。
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CoordinateAttention(nn.Module):
    """
    Coordinate Attention (坐标注意力) 模块
    作用：不仅考虑通道间的关系，还将位置信息（X和Y方向）融合进注意力图。
    优势：对于轻量化网络（如 MobileViT），它能在只增加微乎其微参数量的代价下，
          比传统 SE 模块更好地保留空间特征，非常适合密集预测和精细分类任务。
    """
    def __init__(self, inp, oup, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()  # 原论文使用的是 h-swish，这里使用更主流的 SiLU (Swish)
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        
        n, c, h, w = x.size()
        # 分别对 H 和 W 维度进行一维池化
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
# 融合改进的自定义 MobileViT 模型
# ==========================================

class CustomMobileViT(nn.Module):
    """
    改进版本的 MobileViT。
    为了体现毕设工作量，我们不再直接把原生 timm 模型作为黑盒使用。
    核心改动：
    1. 提取骨干网络 (Features) 与分类头 (Head) 的隔离。
    2. 在骨干网络的末端（即全局池化之前），插入我们自定义的注意力模块（默认 Coordinate Attention）。
    3. 重构分类头，将其从单层 Linear 改为双层 MLP，并加入 Dropout (0.2) 提升防过拟合能力。
    """
    def __init__(self, num_classes=100, pretrained=True, attention_type='coord'):
        super().__init__()
        
        # 1. 加载原生 MobileViT，但丢弃其最后的分类器和池化层，只保留特征提取骨干 (Encoder)
        self.backbone = timm.create_model('mobilevit_xxs', pretrained=pretrained, num_classes=0, global_pool='')
        
        # mobilevit_xxs 输出的最后特征图通道数为 320
        # 如果你未来更换了更复杂的 baseline（例如 mobilevit_s），需要按需修改此处的 320 为实际通道数（可通过 print(x.shape) 测试得知）
        in_features = 320 
        
        # 2. 挂载自定义注意力机制（论文核心卖点之一）
        if attention_type == 'coord':
            self.attention = CoordinateAttention(inp=in_features, oup=in_features)
        elif attention_type == 'se':
            self.attention = SEBlock(channel=in_features)
        else:
            self.attention = nn.Identity() # 不使用任何注意力
            
        # 全局池化层（将 B x C x H x W 压平为 B x C）
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 3. 魔改分类头：加入 MLP 结构与 Dropout（论文卖点之二：提升鲁棒性）
        # 相比原生的单线性层分类器，增加一层隐层 (hidden layer)，使分类边界非线性化能力更强。
        hidden_dim = 128
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2), # 虽然网络有了 LabelSmoothing 和 AutoAugment 等较强的正则手段，Dropout 依然有其独立价值。
            nn.Linear(in_features, hidden_dim),
            nn.SiLU(), # 激活函数
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # [B, 3, 224, 224] -> [B, 320, 7, 7]
        x = self.backbone(x)
        
        # 穿过自定义的注意力模块，重标定特征权重集
        x = self.attention(x)
        
        # [B, 320, 7, 7] -> [B, 320, 1, 1] -> [B, 320]
        x = self.global_pool(x).flatten(1)
        
        # 过最后的定制非线性分类头
        x = self.classifier(x)
        
        return x

if __name__ == '__main__':
    # 测试脚本：确保模型能正常跑通，并观察参数数量
    model = CustomMobileViT(num_classes=100, pretrained=False, attention_type='coord')
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"输出维度: {output.shape}") 
    
    # 统计模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"当前构建的改进版 MobileViT 模型总参数量: {total_params / 1e6:.2f} M")
