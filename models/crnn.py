import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

try:
    from .fusion import ChannelSpatialFusion, PositionalEncoding, RefAwareFusion, HybridAttentionFusion
except ImportError:
    from fusion import ChannelSpatialFusion, PositionalEncoding, RefAwareFusion, HybridAttentionFusion


class MultiFrameCRNN(nn.Module):
    def __init__(self, num_classes, d_model=512):
        super().__init__()
        # Backbone ConvNeXt Tiny
        convnext = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        self.backbone = convnext.features # Output channels: 768

        self.conv_proj = nn.Sequential(
            # Upsample Width by 4x to ensure enough time-steps for CTC
            nn.Upsample(scale_factor=(1, 4), mode='bilinear', align_corners=True),
            nn.Conv2d(768, d_model, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.BatchNorm2d(d_model),
            nn.ReLU()
        )

        self.fusion = HybridAttentionFusion(channels=d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=d_model*4, dropout=0.15, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # Output Layer
        self.fc = nn.Linear(d_model, num_classes)

    def freeze_backbone(self, freeze=True):
        """Freeze or unfreeze the backbone weights."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        print(f"❄️ ConvNeXt Backbone: {'Frozen' if freeze else 'Unfrozen'}")

    def forward(self, x):
        # Input shape: [Batch, Frames, Channels, Height, Width]
        b, t, c, h, w = x.size()

        # Combine Batch and Frames
        x = x.view(b * t, c, h, w)

        # 1. Feature Extraction
        features = self.backbone(x) # [B*T, 768, H/32, W/128]

        # 2. Projection & Upsampling
        features = self.conv_proj(features)

        # 3. Fusion across frames
        fused = self.fusion(features, t)

        # 4. Sequence Modeling
        fused = F.adaptive_avg_pool2d(fused, (1, None))
        seq = fused.squeeze(2).permute(0, 2, 1) # [Batch, Width_New, d_model]

        seq = self.pos_encoder(seq)
        out = self.transformer(seq)
        out = self.fc(out)
        
        return out.log_softmax(2)
