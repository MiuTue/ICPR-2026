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


# ---------------------------------------------------------------------------
# EDSR — rebuilt from the paper:
#   "Enhanced Deep Residual Networks for Single Image Super-Resolution"
#   Lim et al., CVPRW 2017
#   https://arxiv.org/abs/1707.02921
#
# Deviations from paper:
#   - 16 ResBlocks (EDSR-16) instead of 32 (EDSR-32) to fit in GPU memory
#   - residual_scale=0.1 (paper default) to stabilise training with many blocks
#   - No BatchNorm anywhere (paper explicitly removes it)
#   - PixelShuffle ×2 upscale  (paper uses same)
#   - Global skip connection from input to output
# ---------------------------------------------------------------------------

class EDSRResBlock(nn.Module):
    """
    Single EDSR residual block.
    - No BatchNorm (per paper)
    - Optional residual_scale to prevent training instability
    """
    def __init__(self, features: int, residual_scale: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.scale = residual_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv2(self.relu(self.conv1(x)))
        return x + residual * self.scale


class EDSR(nn.Module):
    """
    EDSR-16 super-resolution network.

    Architecture (EDSR-16, scale=2):
        Input [B,3,H,W]
          │ Conv(3→256)
          ▼
          ├────────────────────────────────┐
          │  16 × EDSRResBlock(256)         │  ← residual branch
          │  (no BN, residual_scale=0.1)    │
          └────────────────────────────────┘
          │  Conv(256→256)
          │  PixelShuffle upscale ×2        │  ← 2× spatial upscale
          │  Conv(256→3)
          ▼
        Output [B,3,2H,2W]
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        features: int = 256,       # paper: 256 for EDSR-16
        num_blocks: int = 16,      # EDSR-16 → 16 blocks; use 32 for EDSR-32
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.num_blocks = num_blocks

        # --- Entry convolution (no bias because followed by ReLU — paper convention)
        self.head = nn.Conv2d(in_channels, features, kernel_size=3, padding=1)

        # --- Residual body: 16 EDSRResBlock, NO BatchNorm
        self.body = nn.Sequential(*[
            EDSRResBlock(features, residual_scale)
            for _ in range(num_blocks)
        ])

        # --- Bridge convolution before upscale
        self.bypass = nn.Conv2d(features, features, kernel_size=3, padding=1)

        # --- 2× upscale: PixelShuffle
        self.upscale = nn.Sequential(
            nn.Conv2d(features, features * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
        )

        # --- Output convolution
        self.tail = nn.Conv2d(features, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global skip connection (paper: residual on the identity)
        x = self.head(x)
        x = x + self.body(x)        # residual on the output of head
        x = self.bypass(x)
        x = self.upscale(x)
        x = self.tail(x)
        return x


# ---------------------------------------------------------------------------
# Legacy alias — SRModule is now a thin wrapper around EDSR
# ---------------------------------------------------------------------------
class SRModule(EDSR):
    """Legacy alias; all new code should use EDSR directly."""
    def __init__(self, in_channels=3, out_channels=3, features=64, num_blocks=4):
        # Map old parameters to EDSR naming (features was 64, now 256)
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            features=256,
            num_blocks=16,
            residual_scale=0.1,
        )


class EndToEndLPR(nn.Module):
    """
    End-to-end pipeline: EDSR-16 Super-Resolution + Swin-Tiny + Transformer Encoder.

    Data-flow (32×128 → 64×256):
        LR [B,3,32,128]
          │
          ▼  EDSR-16 (21.8 M params, features=256, 16 ResBlocks)
        SR [B,3,64,256]
          │
          ▼  Swin-Tiny backbone (frozen, pretrained)
        F  [B,2,8,768]              ← Swin output: H'=2, W'=8, C=768
          │
          ▼  ConvProj: [B,2,8,768] → permute → [B,768,2,8]
                              → Upsample W×4: [B,768,2,32]
                              → Conv(768→512) + AvgPool H→1: [B,512,1,32]
                              → squeeze → [B,32,512]
        Seq [B,T=32,d_model=512]
          │
          ▼  PositionalEncoding + TransformerEncoder(8 layers, 8 heads)
        logits [B,32,512]
          │
          ▼  FC + LogSoftmax → [B,32,NUM_CLASSES]  (CTC blank=0)
    """
    def __init__(self, num_classes, d_model=512):
        super().__init__()
        # 1. Super-Resolution: EDSR-16
        self.sr_module = EDSR()

        # 2. Recognition: Swin-Tiny backbone (frozen pretrained)
        swin = swin_t(weights=Swin_T_Weights.DEFAULT)
        self.backbone = swin.features

        # 3. ConvProj: Swin output [B,2,8,768] → [B,512,1,32]
        #    Width: 8 → 32 (×4 via Upsample) to get enough CTC timesteps
        self.conv_proj = nn.Sequential(
            nn.Upsample(scale_factor=(1, 4), mode='bilinear', align_corners=False),
            nn.Conv2d(768, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None)),  # height → 1
        )

        self.pos_encoder = PositionalEncoding(d_model=d_model)

        # 4. Transformer Encoder (8 layers, 8 heads)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=8)

        # 5. Output: FC → LogSoftmax (CTC blank index = 0)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor):
        # x: [B, 3, 32, 128]
        sr_img = self.sr_module(x)                         # [B, 3, 64, 256]
        features = self.backbone(sr_img)                    # [B, 2, 8, 768]

        # ConvProj: [B,2,8,768] → [B,512,1,32]
        features = features.permute(0, 3, 1, 2)            # [B, 768, 2, 8]
        features = self.conv_proj(features)                # [B, 512, 1, 32]

        # Prepare sequence: [B, T=32, d_model=512]
        seq = features.squeeze(2).permute(0, 2, 1)         # [B, 32, 512]

        seq = self.pos_encoder(seq)
        out = self.transformer(seq)                          # [B, 32, 512]
        out = self.fc(out)                                   # [B, 32, NUM_CLASSES]

        return sr_img, out.log_softmax(2)                     # [B, 32, NUM_CLASSES]
