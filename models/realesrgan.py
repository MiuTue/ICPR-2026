"""
Real-ESRGAN Generator — implemented from scratch.

Paper: "Real-ESRGAN: Training Real-World Blind Super-Resolution with
        Pure Synthetic Data"
        Wang et al., ICCV 2021 Workshop
        https://arxiv.org/abs/2107.10833

Architecture (Real-ESRNet / Real-ESRGAN generator):
    Input LR [B,3,H,W]
      │
      │  Conv(3→64)
      ▼
      │  ┌──────────────────────────────────┐
      │  │  num_rrdb × RRDB(64)             │  ← Residual-in-Residual Dense Block
      │  │  (with residual_scaling=0.2)      │
      │  └──────────────────────────────────┘
      │  +
      │  Conv(64→64)
      │  ┌──────────────────────────────────┐
      │  │  PixelShuffle ×2 upscale         │
      │  └──────────────────────────────────┘
      │  ┌──────────────────────────────────┐
      │  │  PixelShuffle ×2 upscale (×4 SR) │
      │  └──────────────────────────────────┘
      │  Conv(64→64)
      │  Conv(64→3)
      ▼
    Output HR [B,3,4H,4W]

Deviations from paper:
    - 6 RRDB blocks (vs 23 in paper) to keep GPU memory manageable
      for joint SR+recognition training
    - Scale fixed at ×4 (paper supports ×2/×3/×4)
    - No spectral normalization (used only for discriminator GAN training)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper layers
# ---------------------------------------------------------------------------

class DenseLayer(nn.Module):
    """Dense connection layer (1×1 conv that concatenates all previous features)."""

    def __init__(self, channels: int, growth_channels: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(channels, growth_channels, kernel_size=3, padding=1)
        # negative_slope=0.2 là chuẩn cho các model GAN học thuật
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lrelu(self.conv(x))
        
        return torch.cat([x, self.conv(x)], dim=1)


class RRDB(nn.Module):
    """
    Residual-in-Residual Dense Block.

    Architecture:
        x ──────────────────────────────────┐
        │                                   │
        ▼                                   │
        DenseLayer → DenseLayer → DenseLayer│  (grows channels by 3×growth)
        │                                   │
        ▼                                   │
        Conv(channels + 3*growth → channels)│
        │                                   │
        x ← ────────────────────────────────┘  residual (scaled)
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
            DenseLayer(channels, growth_channels),                       # C -> C+G
            DenseLayer(channels + growth_channels, growth_channels),      # C+G -> C+2G
            DenseLayer(channels + 2 * growth_channels, growth_channels),  # C+2G -> C+3G
            DenseLayer(channels + 3 * growth_channels, growth_channels),  # C+3G -> C+4G
            # Lớp cuối cùng nén tất cả về lại channels ban đầu
            nn.Conv2d(channels + 4 * growth_channels, channels, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection
        return x + self.dense_blocks(x) * self.residual_scaling


# ---------------------------------------------------------------------------
# Upsampler helpers (paper uses pixel shuffle ×2 twice for ×4 SR)
# ---------------------------------------------------------------------------

class UpsampleBlock(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.upsample = nn.Sequential(
            # --- Bước 1: Phóng to 2x (H, W -> 2H, 2W) ---
            nn.Conv2d(features, features * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # --- Lớp Conv trung gian (Bảo toàn features và H, W) ---
            # Giúp tinh chỉnh đặc trưng trước khi phóng lần 2
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # --- Bước 2: Phóng to 2x tiếp (2H, 2W -> 4H, 4W) ---
            nn.Conv2d(features, features * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(x)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class RealESRGANGenerator(nn.Module):
    """
    Real-ESRGAN Generator (×4 super-resolution).

    Default: 6 RRDB blocks, 64 features.
    For full paper model: num_rrdb=23, features=64.
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
        assert upscale_factor in (2, 4), "Real-ESRGAN supports ×2 or ×4 upscale only"
        self.upscale_factor = upscale_factor

        # --- Entry convolution
        self.conv_first = nn.Conv2d(in_channels, features, kernel_size=3, padding=1)

        # --- Trunk: num_rrdb × RRDB blocks
        self.trunk = nn.Sequential(*[
            RRDB(features, growth_channels, residual_scaling)
            for _ in range(num_rrdb)
        ])

        # --- Post-RRDB convolution
        self.conv_body = nn.Conv2d(features, features, kernel_size=3, padding=1)

        # --- Upsampling: ×4 total
        if upscale_factor == 4:
            self.upsampler = nn.Sequential(
                UpsampleBlock(features, scale=2),
                UpsampleBlock(features, scale=2),
            )
        else:
            self.upsampler = UpsampleBlock(features, scale=upscale_factor)

        # --- Output convolution
        self.conv_last = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.Conv2d(features, out_channels, kernel_size=3, padding=1),
        )

        # Global residual shortcut
        self._pixel_unshuffle = nn.PixelUnshuffle(upscale_factor) if upscale_factor > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shortcut: average-pooled LR features
        feat = self.conv_first(x)
        trunk = self.conv_body(self.trunk(feat))
        feat = feat + trunk  # global residual
        feat = self.upsampler(feat)
        out = self.conv_last(feat)
        return out

    def get_model_scaling(self) -> tuple[int, int]:
        return (32, 128), (32 * self.upscale_factor, 128 * self.upscale_factor)


# ---------------------------------------------------------------------------
# Lightweight version for memory-efficient joint training
# Uses fewer RRDB blocks and lower feature dim
# ---------------------------------------------------------------------------

class RealESRGANGeneratorLite(nn.Module):
    """
    Lightweight Real-ESRGAN for joint SR + recognition.

    Compared to full model:
        features:    64  → 32
        num_rrdb:    6   → 3
        growth_ch:   32  → 16
        residual_scaling: 0.2 → 0.1

    Output: ×4 upscale (32×128 → 128×512).
    """

    def __init__(self, upscale_factor: int = 4):
        super().__init__()
        self.upscale_factor = upscale_factor

        self.conv_first = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.trunk = nn.Sequential(*[
            RRDB(32, growth_channels=16, residual_scaling=0.1)
            for _ in range(3)
        ])
        self.conv_body = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        if upscale_factor == 4:
            self.upsampler = nn.Sequential(
                UpsampleBlock(32, scale=2),
                UpsampleBlock(32, scale=2),
            )
        else:
            self.upsampler = UpsampleBlock(32, scale=upscale_factor)

        self.conv_last = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        trunk = self.conv_body(self.trunk(feat))
        feat = feat + trunk
        feat = self.upsampler(feat)
        return self.conv_last(feat)

    def get_model_scaling(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return (32, 128), (32 * self.upscale_factor, 128 * self.upscale_factor)
