import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
    """Dense connection with growth rate k."""
    def __init__(self, in_channels, growth_rate=32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return torch.cat([x, self.layers(x)], dim=1)


class ResidualDenseBlock(nn.Module):
    """
    Residual-in-Residual Dense Block (RRDB) used in Real-ESRGAN.
    Three cascaded dense blocks with a residual scaling factor.
    """
    def __init__(self, channels, growth_rate=32, num_layers=3, residual_beta=0.2):
        super().__init__()
        self.residual_beta = residual_beta

        dense_layers = []
        _in = channels
        for _ in range(num_layers):
            dense_layers.append(DenseLayer(_in, growth_rate))
            _in += growth_rate
        self.dense_layers = nn.Sequential(*dense_layers)

        self.conv = nn.Conv2d(_in, channels, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        out = self.dense_layers(x)
        out = self.conv(out)
        return identity + out * self.residual_beta


class RealESRGANUpsampler(nn.Module):
    """
    Lightweight Real-ESRGAN-style upsampler.

    Architecture:
    1. Shallow feature extraction (3 → 64 channels)
    2. 2 × RRDB blocks (residual in residual dense blocks)
    3. Feature fusion + 2× PixelShuffle upscaling
    4. Skip connection from upsampled input for sharp detail

    Input:  [B*T, 3, H, W]
    Output: [B*T, 3, H*2, W*2]
    """
    def __init__(self, in_channels=3, channels=64, growth_rate=32, num_rrdb=2, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

        # Shallow feature extraction
        self.conv_first = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        # Body: stacked RRDB blocks
        rrdb_blocks = []
        for _ in range(num_rrdb):
            rrdb_blocks.append(
                ResidualDenseBlock(channels, growth_rate, num_layers=3, residual_beta=0.2)
            )
        self.rrdb_body = nn.Sequential(*rrdb_blocks)

        # Fusion conv before upscaling
        self.conv_body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        # Upscale: PixelShuffle
        self.upconv = nn.Sequential(
            nn.Conv2d(channels, channels * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.LeakyReLU(0.2, True),
        )

        # Final reconstruction conv
        self.conv_last = nn.Conv2d(channels, in_channels, kernel_size=3, padding=1)

        # Skip connection: upscale input directly and combine
        self.skip_conv = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        """
        Args:
            x: [B*T, 3, H, W]
        Returns:
            upsampled: [B*T, 3, H*2, W*2]
        """
        # Shallow features
        feat = self.conv_first(x)                        # [B*T, 64, H, W]

        # RRDB body
        body_feat = self.rrdb_body(feat)                  # [B*T, 64, H, W]
        body_feat = self.conv_body(body_feat)            # [B*T, 64, H, W]

        # Fusion: residual from initial features
        feat = feat + body_feat                          # [B*T, 64, H, W]

        # Upscale
        upscaled = self.upconv(feat)                     # [B*T, 64, H*2, W*2]

        # Skip connection: bilinearly upsampled input
        x_up = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear',
                             align_corners=True)          # [B*T, 3, H*2, W*2]
        skip = self.skip_conv(x_up)                      # [B*T, 64, H*2, W*2]
        upscaled = upscaled + skip                        # [B*T, 64, H*2, W*2]

        # Final reconstruction
        out = self.conv_last(upscaled)                   # [B*T, 3, H*2, W*2]
        return out
