import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_t, Swin_T_Weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ResBlock(nn.Module):
    """Residual block for EDSR-based Super Resolution module."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))


class SRModule(nn.Module):
    """Dedicated Super Resolution Module (EDSR style)."""
    def __init__(self, in_channels=3, out_channels=3, features=64, num_blocks=4):
        super().__init__()
        self.head = nn.Conv2d(in_channels, features, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[ResBlock(features) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(features, features, kernel_size=3, padding=1)
        
        # 2x Upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(features, features * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(features, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        h = self.head(x)
        b = self.body(h)
        b = self.tail(b)
        h = h + b
        out = self.upsample(h)
        return out


class EndToEndLPR(nn.Module):
    """End-to-end pipeline: SR Module + ConvNet/Transformer Recognition."""
    def __init__(self, num_classes, d_model=512):
        super().__init__()
        # 1. Super-Resolution Module
        self.sr_module = SRModule()

        # 2. Recognition Module
        # Backbone Swin Transformer
        swin = swin_t(weights=Swin_T_Weights.DEFAULT)
        self.backbone = swin.features

        self.conv_proj = nn.Sequential(
            # Upsample Width by 4x to ensure enough time-steps for CTC
            nn.Upsample(scale_factor=(1, 4), mode='bilinear', align_corners=True),
            # Downsample Height to 1
            nn.Conv2d(768, d_model, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.BatchNorm2d(d_model),
            nn.ReLU()
        )

        self.pos_encoder = PositionalEncoding(d_model=d_model)

        # Sequence Modeling (Transformer Encoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=d_model*4, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=8)

        # Output Layer
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x is LR image: [Batch, 3, 32, 128]
        
        # 1. Super-Resolution (Upscales to 64x256)
        sr_img = self.sr_module(x)
        
        # 2. Feature Extraction
        # Swin expects 224x224, but works dynamically. 64/32=2, 256/32=8.
        features = self.backbone(sr_img) # [B, 2, 8, 768]
        features = features.permute(0, 3, 1, 2) # [B, 768, 2, 8]

        # 3. Projection & Upsampling
        features = self.conv_proj(features) # [B, 512, 1, 32]

        # 4. Sequence Modeling
        features = F.adaptive_avg_pool2d(features, (1, None))
        seq = features.squeeze(2).permute(0, 2, 1) # [Batch, T, d_model]

        seq = self.pos_encoder(seq)
        out = self.transformer(seq)
        out = self.fc(out)
        
        # Return both the super-resolved image (for SR loss) and the CTC predictions
        return sr_img, out.log_softmax(2)
