import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_t, Swin_T_Weights

try:
    from .fusion import ChannelSpatialFusion, PositionalEncoding, RefAwareFusion
except ImportError:
    from fusion import ChannelSpatialFusion, PositionalEncoding, RefAwareFusion


class MultiFrameCRNN(nn.Module):
    def __init__(self, num_classes, d_model=512):
        super().__init__()
        # Backbone Swin Transformer (Tiny version for speed)
        swin = swin_t(weights=Swin_T_Weights.DEFAULT)
        self.backbone = swin.features # Output channels: 768

        self.conv_proj = nn.Sequential(
            # Upsample Width by 4x to ensure enough time-steps for CTC
            nn.Upsample(scale_factor=(1, 4), mode='bilinear', align_corners=True),
            nn.Conv2d(768, d_model, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.BatchNorm2d(d_model),
            nn.ReLU()
        )

        self.fusion = RefAwareFusion(channels=d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=d_model*4, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=8)

        # Output Layer
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # Input shape: [Batch, Frames, Channels, Height, Width]
        b, t, c, h, w = x.size()

        # Combine Batch and Frames
        x = x.view(b * t, c, h, w)

        # 1. Feature Extraction
        features = self.backbone(x) # Swin Tiny features: [B*T, H/32, W/32, 768]
        features = features.permute(0, 3, 1, 2) # Convert to [B*T, 768, H/32, W/128] for Conv layers

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
