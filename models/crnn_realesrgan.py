"""
End-to-End LPR pipeline with Real-ESRGAN SR + ConvNeXt-Tiny backbone.
Variant of models/crnn.py replacing EDSR-16 -> Real-ESRGAN and Swin-T -> ConvNeXt-Tiny.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


# ============================================================================
# Positional Encoding
# ============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        return x + self.pe[:, :x.size(1)]


# ============================================================================
# Real-ESRGAN building blocks
# ============================================================================

class DenseLayer(nn.Module):
    """
    Dense connection layer: convolves on concatenated [x, new_features].

    Grows channels by growth_channels each forward pass.
    Uses LeakyReLU(0.2) per Real-ESRGAN paper convention.
    """
    def __init__(self, channels: int, growth_channels: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(channels, growth_channels, kernel_size=3, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, C+growth, H, W]
        return torch.cat([x, self.lrelu(self.conv(x))], dim=1)


class RRDB(nn.Module):
    """
    Residual-in-Residual Dense Block (Real-ESRGAN).

    Architecture per block:
        x ------------------------------------------------------+
        |                                                       |
        v                                                       |
        DenseLayer  ->  C       -> C+G                          |
        v                                                       |
        DenseLayer  ->  C+G     -> C+2G                         |
        v                                                       |
        DenseLayer  ->  C+2G    -> C+3G                         |
        v                                                       |
        DenseLayer  ->  C+3G    -> C+4G                         |
        v                                                       |
        Conv(C+4G->C)  (compress back to C)                      |
        |                                                       |
        x <-----------------------------------------------------+  residual (scaled x 0.2)
    """
    def __init__(
        self,
        channels: int,
        growth_channels: int = 32,
        residual_scaling: float = 0.2,
    ):
        super().__init__()
        self.residual_scaling = residual_scaling

        self.dense_blocks = nn.Sequential(
            DenseLayer(channels,                              growth_channels),   # C   -> C+G
            DenseLayer(channels + growth_channels,            growth_channels),  # C+G -> C+2G
            DenseLayer(channels + 2 * growth_channels,        growth_channels),  # C+2G-> C+3G
            DenseLayer(channels + 3 * growth_channels,        growth_channels),  # C+3G-> C+4G
            nn.Conv2d(channels + 4 * growth_channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, C, H, W] (same shape, residual connection)
        return x + self.dense_blocks(x) * self.residual_scaling


class UpsampleBlock(nn.Module):
    """
    Upsample by x4 with intermediate refinement convs.

    Per Real-ESRGAN paper: x2 upscale -> conv refine -> x2 upscale -> conv refine.

    Channel flow (features=64):
        Input  [B, 64, H,  W]
          | Conv(64->256) + PixelShuffle -> [B, 64, 2H,  2W]
          | Conv(64->64)  (no spatial change) -> [B, 64, 2H,  2W]
          | Conv(64->256) + PixelShuffle -> [B, 64, 4H,  4W]
          | Conv(64->64)  (no spatial change) -> [B, 64, 4H,  4W]
        Output [B, 64, 4H, 4W]  (x4 spatial upscale total)
    """
    def __init__(self, features: int):
        super().__init__()
        self.upsample = nn.Sequential(
            # --- Step 1: x2 upscale + refine ---
            nn.Conv2d(features, features * 4, kernel_size=3, padding=1),  # C->4C
            nn.PixelShuffle(2),                                              # [B,4C,H,W]->[B,C,2H,2W]
            nn.LeakyReLU(0.2, inplace=True),
            # --- Refinement conv (preserve C, H, W) ---
            nn.Conv2d(features, features, kernel_size=3, padding=1),        # C->C, [B,C,2H,2W]
            nn.LeakyReLU(0.2, inplace=True),
            # --- Step 2: x2 upscale + refine ---
            nn.Conv2d(features, features * 4, kernel_size=3, padding=1),  # C->4C
            nn.PixelShuffle(2),                                              # [B,4C,2H,2W]->[B,C,4H,4W]
            nn.LeakyReLU(0.2, inplace=True),
            # --- Refinement conv (preserve C, H, W) ---
            nn.Conv2d(features, features, kernel_size=3, padding=1),        # C->C, [B,C,4H,4W]
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, C, 4H, 4W]
        return self.upsample(x)


class RealESRGANGenerator(nn.Module):
    """
    Real-ESRGAN Generator (x4 super-resolution).

    Input  [B, 3, 32, 128]
      | Conv(3->64)
      v
      |  6 x RRDB(64, G=32, scale=0.2)
      |  Conv(64->64)
      |  +
      |  UpsampleBlock(64)  x4 spatial
      |  Conv(64->64) + Conv(64->3)
      v
    Output [B, 3, 128, 512]

    Total: ~1.42 M params
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        features: int = 64,
        num_rrdb: int = 6,
        growth_channels: int = 32,
        residual_scaling: float = 0.2,
        upscale_factor: int = 4,
    ):
        super().__init__()
        self.upscale_factor = upscale_factor

        # --- Entry ---
        # [B, 3, H, W] -> [B, 64, H, W]
        self.conv_first = nn.Conv2d(in_channels, features, kernel_size=3, padding=1)

        # --- Trunk: num_rrdb x RRDB ---
        # [B, 64, H, W] -> [B, 64, H, W]
        self.trunk = nn.Sequential(*[
            RRDB(features, growth_channels, residual_scaling)
            for _ in range(num_rrdb)
        ])

        # --- Post-RRDB conv ---
        # [B, 64, H, W] -> [B, 64, H, W]
        self.conv_body = nn.Conv2d(features, features, kernel_size=3, padding=1)

        # --- Upsampling: x4 total (one UpsampleBlock = x4 upscale) ---
        # [B, 64, H, W] -> UpsampleBlock -> [B, 64, 4H, 4W]
        self.upsampler = UpsampleBlock(features)

        # --- Output ---
        # [B, 64, 4H, 4W] -> [B, 64, 4H, 4W] -> [B, 3, 4H, 4W]
        self.conv_last = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.Conv2d(features, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, 3, 32, 128]
        feat = self.conv_first(x)                    # [B, 64, 32, 128]
        trunk = self.conv_body(self.trunk(feat))     # [B, 64, 32, 128]
        feat = feat + trunk                          # [B, 64, 32, 128]  (global residual)
        feat = self.upsampler(feat)                  # [B, 64, 128, 512] (x4 upscale)
        out = self.conv_last(feat)                   # [B, 3, 128, 512]
        return out


# ============================================================================
# ConvNeXt-Tiny backbone wrapper
# ============================================================================

class ConvNeXtBackbone(nn.Module):
    """
    ConvNeXt-Tiny frozen pretrained backbone.

    Spatial output (H', W') scales with input via 4 downsampling stages (stride 2 each):
        32x128  -> [B, 768,  1,   4]
        64x256  -> [B, 768,  2,   8]
        128x512 -> [B, 768,  4,  16]   <- RealESRGAN output feeds this
        256x1024-> [B, 768,  8,  32]

    ConvNeXt-Tiny: 28.59 M params (vs Swin-T: 28.29 M params).
    Final stage channels: 768.
    """
    def __init__(self):
        super().__init__()
        convnext = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        self.features = convnext.features  # Sequential
        # No classifier/head needed; just the feature extractor.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, 3, H, W] -> [B, 768, H/32, W/32]
        return self.features(x)


# ============================================================================
# Conv Projection (compatible with ConvNeXt output [B, 768, H', W'])
# ============================================================================

class ConvProj(nn.Module):
    """
    Maps ConvNeXt features -> Transformer sequence [B, T, d_model=512].

    With RealESRGAN output (128x512) -> ConvNeXt -> [B, 768, 4, 16]:
        [B, 768, 4, 16]          (ConvNeXt output for 128x512 input)
          | Conv(768->512): 3x3    -> [B, 512, 4, 16]
          | BatchNorm + ReLU
          | AdaptiveAvgPool H->1   -> [B, 512, 1, 16]
          | squeeze(2)             -> [B, 512, 16]
          | permute(0,2,1)         -> [B, 16, 512]
        Output: [B, T=16, d_model=512]

    NOTE: ConvNeXt output spatial size depends on input resolution.
          This ConvProj handles any H', W' via AdaptiveAvgPool2d((1, None)).
    """
    def __init__(self, in_channels: int = 768, d_model: int = 512):
        super().__init__()
        self.proj = nn.Sequential(
            # --- Channel compression + refinement ---
            nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1),
            # [B, 768, H', W'] -> [B, 512, H', W']
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            # --- Spatial: pool H to 1, keep W as CTC timesteps ---
            nn.AdaptiveAvgPool2d((1, None)),
            # [B, 512, H', W'] -> [B, 512, 1, W']
        )
        self.T = None  # set dynamically from input in forward()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 768, H', W'] (e.g. [B, 768, 4, 16] for 128x512 input)
        x = self.proj(x)                     # [B, 512, 1, W']
        x = x.squeeze(2).permute(0, 2, 1)  # [B, W', 512]
        return x


# ============================================================================
# End-to-End LPR with Real-ESRGAN + ConvNeXt-Tiny
# ============================================================================

class EndToEndLPR(nn.Module):
    """
    Full pipeline: Real-ESRGAN (x4) -> ConvNeXt-Tiny -> Transformer -> CTC.

    Input:  LR image [B, 3, 32, 128]
    Output: SR image [B, 3, 128, 512] + CTC logits [B, T=16, NUM_CLASSES]

    Shape trace:
        [B, 3,  32,  128]          -- RealESRGAN (x4 upscale) --
        -> [B, 3,  128, 512]        SR image (for MSELoss)
        -> [B, 768,   4,  16]       ConvNeXt-Tiny backbone
        -> [B, 512,   1,  16]       Conv(768->512) + BN + ReLU + AvgPool H->1
        -> [B, 512,  16]            squeeze dim 2
        -> [B,  16, 512]            permute (T=16, d_model=512)
        -> [B,  16, 512]            PositionalEncoding + TransformerEncoder
        -> [B,  16, NUM_CLASSES]   FC + LogSoftmax (CTC blank=0)
    """
    def __init__(self, num_classes: int, d_model: int = 512):
        super().__init__()

        # 1. Super-Resolution: Real-ESRGAN (x4 upscale)
        self.sr_module = RealESRGANGenerator(
            in_channels=3,
            out_channels=3,
            features=64,
            num_rrdb=6,
            growth_channels=32,
            residual_scaling=0.2,
            upscale_factor=4,
        )

        # 2. Recognition: ConvNeXt-Tiny backbone (frozen pretrained)
        self.backbone = ConvNeXtBackbone()

        # 3. ConvProj: [B,768,H',W'] -> [B,T,d_model=512]
        self.conv_proj = ConvProj(in_channels=768, d_model=d_model)

        # 4. Positional encoding + Transformer Encoder
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,                   # 512/4=128 per head; 4 heads
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # 5. Output: FC -> LogSoftmax (CTC blank index = 0)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor):
        # x: [B, 3, 32, 128]
        sr_img   = self.sr_module(x)              # [B,  3, 128, 512]  (SR image for MSE)
        features = self.backbone(sr_img)          # [B, 768,   4,  16]  (ConvNeXt features)
        seq      = self.conv_proj(features)       # [B,  16, 512]       (T=16 timesteps)
        seq      = self.pos_encoder(seq)          # [B,  16, 512]       (+ positional)
        out      = self.transformer(seq)          # [B,  16, 512]       (4-layer encoder)
        logits   = self.fc(out).log_softmax(2)    # [B,  16, NUM_CLASSES]

        return sr_img, logits  # (SR for MSE loss, CTC logits for recognition)
