import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

try:
    from .fusion import HybridAttentionFusion
    from .stn import STN, STNWithUpsampler
    from .realesrgan import RealESRGANUpsampler
except ImportError:
    from fusion import HybridAttentionFusion
    from stn import STN, STNWithUpsampler
    from realesrgan import RealESRGANUpsampler


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: [Batch, Seq_Len, d_model]
        return x + self.pe[:, :x.size(1)]


class MultiFrameCRNN(nn.Module):
    """
    Multi-Frame CRNN with STN + RealESRGAN super-resolution.

    Pipeline:
        Input [B, T, 3, 64, 256]
            ↓
        STN (geometric correction)  →  [B*T, 3, 64, 256]
            ↓
        RealESRGAN ×1 (2× up)        →  [B*T, 3, 128, 512]
            ↓
        RealESRGAN ×2 (2× up)        →  [B*T, 3, 256, 1024]
            ↓
        ConvNeXt Tiny Backbone       →  [B*T, 768, 8, 32]
            ↓
        ConvProj (768→512, H:8→1)    →  [B*T, 512, 1, 32]
            ↓
        HybridAttentionFusion        →  [B, 512, 1, 32]
            ↓
        AdaptiveAvgPool + Permute    →  [B, 32, 512]
            ↓
        PositionalEncoding + TransformerEncoder(6)
            ↓
        FC + log_softmax             →  [B, 32, num_classes]
    """
    def __init__(self, num_classes, d_model=512, use_stn=True, use_sr=True):
        super().__init__()
        self.use_stn = use_stn
        self.use_sr = use_sr

        # 1. STN — Spatial Transformer for geometric correction
        if use_stn:
            self.stn = STN(in_channels=3)

        # 2. RealESRGAN ×2 for 4× super-resolution
        if use_sr:
            # After 2×: 64 → 128
            self.sr_up1 = RealESRGANUpsampler(
                in_channels=3, channels=64, growth_rate=32,
                num_rrdb=2, scale_factor=2
            )
            # After another 2×: 128 → 256
            self.sr_up2 = RealESRGANUpsampler(
                in_channels=3, channels=64, growth_rate=32,
                num_rrdb=2, scale_factor=2
            )

        # 3. Backbone ConvNeXt Tiny
        convnext = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        self.backbone = convnext.features  # Output channels: 768

        # 4. Projection → bring to d_model channels, reduce H to 1
        self.conv_proj = nn.Sequential(
            nn.Conv2d(768, d_model, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.BatchNorm2d(d_model),
            nn.ReLU()
        )

        # 5. Temporal Fusion
        self.fusion = HybridAttentionFusion(channels=d_model)

        # 6. Sequence Modeling
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=d_model * 4,
            dropout=0.15, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # 7. Output
        self.fc = nn.Linear(d_model, num_classes)

    def freeze_backbone(self, freeze=True):
        """Freeze or unfreeze the backbone weights."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        print(f"❄️ ConvNeXt Backbone: {'Frozen' if freeze else 'Unfrozen'}")

    def freeze_sr(self, freeze=True):
        """Freeze or unfreeze the SR + STN weights."""
        if self.use_sr:
            for param in self.sr_up1.parameters():
                param.requires_grad = not freeze
            for param in self.sr_up2.parameters():
                param.requires_grad = not freeze
        if self.use_stn:
            for param in self.stn.parameters():
                param.requires_grad = not freeze
        print(f"🔧 SR+STN modules: {'Frozen' if freeze else 'Unfrozen'}")

    def forward(self, x):
        # Input shape: [Batch, Frames, Channels, Height, Width]
        b, t, c, h, w = x.size()

        # Combine Batch and Frames
        x = x.view(b * t, c, h, w)  # [B*T, 3, H, W]

        # 1. STN: geometric correction
        if self.use_stn:
            x = self.stn(x)  # [B*T, 3, H, W]

        # 2. RealESRGAN: 2× super-resolution (4× total)
        if self.use_sr:
            x = self.sr_up1(x)  # [B*T, 3, H*2, W*2]
            x = self.sr_up2(x)  # [B*T, 3, H*4, W*4]

        # 3. Feature extraction through backbone
        # For input 64×256, after SR 256×1024:
        #   ConvNeXt: stride 32 → 256/32=8, 1024/32=32
        features = self.backbone(x)  # [B*T, 768, H/32, W/32]

        # 4. Projection
        features = self.conv_proj(features)  # [B*T, 512, H/64, W/32]

        # 5. Temporal fusion across frames
        fused = self.fusion(features, t)  # [B, 512, H/64, W/32]

        # 6. Sequence preparation
        fused = F.adaptive_avg_pool2d(fused, (1, None))
        seq = fused.squeeze(2).permute(0, 2, 1)  # [B, W/32, 512]

        # 7. Transformer sequence modeling
        seq = self.pos_encoder(seq)
        out = self.transformer(seq)
        out = self.fc(out)

        return out.log_softmax(2)
