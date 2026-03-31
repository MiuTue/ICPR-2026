"""
Module 1: Lightweight Super-Resolution Upsampler.

Upscales LR image (64×256) → SR image (256×1024) = 4× scale.

Architecture:
    Input [B*T, 3, 64, 256]
        → Shallow feature (3→64)
        → Residual blocks (2×)
        → Upsample (2×)  → [B*T, 3, 128, 512]
        → Upsample (2×)  → [B*T, 3, 256, 1024]
        → Output

Giữ concept "học upscaling" thay vì chỉ interpolate,
nhưng nhẹ hơn RealESRGAN rất nhiều.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Single residual block: Conv → BN → ReLU → Conv → BN → residual."""
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.net(x)


class LightSR(nn.Module):
    """
    Lightweight SR module — upsamples LR → HR (4× total).

    Input:  [B*T, 3, 64, 256]
    Output: [B*T, 3, 256, 1024]

    Two 2× upsampling stages, each with residual blocks.
    Final output is a residual connection from bilinearly-upsampled input.
    """
    def __init__(self, in_channels=3, hidden_channels=64, num_res_blocks=2):
        super().__init__()

        # Shallow feature extraction
        self.shallow = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(True),
        )

        # Residual body
        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(ResidualBlock(hidden_channels))
        self.res_body = nn.Sequential(*res_blocks)

        # Channel fusion before upsampling
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(True),
        )

        # 1st Upsample: 64×256 → 128×512 (2×)
        self.up1 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(True),
        )

        # 2nd Upsample: 128×512 → 256×1024 (2×)
        self.up2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(True),
        )

        # Final reconstruction
        self.output = nn.Conv2d(hidden_channels, in_channels, 3, padding=1)

    def forward(self, x):
        # Shallow features
        feat = self.shallow(x)          # [B*T, 64, H, W]

        # Residual body
        feat = self.res_body(feat)      # [B*T, 64, H, W]
        feat = self.fusion(feat)        # [B*T, 64, H, W]

        # Upsample 2×
        feat = self.up1(feat)            # [B*T, 64, H*2, W*2]

        # Upsample 2× tiếp → total 4×
        feat = self.up2(feat)            # [B*T, 64, H*4, W*4]

        # Bilinear upscale input để residual
        x_up = F.interpolate(x, scale_factor=4, mode='bilinear',
                             align_corners=True)  # [B*T, 3, H*4, W*4]

        # Residual connection: learned detail + bilinear baseline
        out = self.output(feat)          # [B*T, 3, H*4, W*4]
        return out + x_up
