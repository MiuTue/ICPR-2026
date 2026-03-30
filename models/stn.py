import torch
import torch.nn as nn
import torch.nn.functional as F


class STN(nn.Module):
    """
    Spatial Transformer Network (STN).

    Applies an affine transformation to each input frame to correct
    geometric distortions (slight rotations, scale, translation).
    The transformation is predicted per-frame from pooled features.
    """
    def __init__(self, in_channels=3, grid_size=5):
        super().__init__()
        self.grid_size = grid_size

        # Localization network: predicts 6 affine parameters (θ)
        # θ = [scale_x, scale_y, shift_x, shift_y, rotation, shear] → simplified to [a,b,c,d,e,f]
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 6)  # 6 parameters for affine transform
        )
        # Initialize the identity transformation
        nn.init.zeros_(self.fc_loc[-1].weight)
        nn.init.zeros_(self.fc_loc[-1].bias)
        # bias = [1, 0, 0, 0, 1, 0]  → identity affine

    def forward(self, x):
        """
        Args:
            x: [B*T, C, H, W] input frames
        Returns:
            transformed: [B*T, C, H, W] geometrically corrected frames
        """
        btc = x.size(0)

        # 1. Predict affine parameters θ
        feat = self.localization(x)                      # [B*T, 128, 1, 1]
        feat = feat.view(feat.size(0), -1)               # [B*T, 128]
        theta = self.fc_loc(feat)                        # [B*T, 6]

        # Scale translation to reasonable magnitude
        theta = torch.tanh(theta)                         # constrain to [-1, 1]

        # Build affine transformation matrix [B*T, 2, 3]
        # Flat [6] → 2×3: [[a,b,c],[d,e,f]]
        theta_matrix = theta.view(-1, 2, 3)               # [B*T, 2, 3]

        # 2. Generate sampling grid
        batch_size = btc
        H, W = x.shape[2], x.shape[3]
        grid = F.affine_grid(
            theta_matrix, x.size(), align_corners=True
        )  # [B*T, H, W, 2]

        # 3. Sample using bilinear interpolation
        transformed = F.grid_sample(
            x, grid, mode='bilinear', padding_mode='border', align_corners=True
        )
        return transformed


class STNWithUpsampler(nn.Module):
    """
    STN followed by a lightweight upsampler for super-resolution.
    Used to boost LR frame resolution before the main backbone.
    """
    def __init__(self, in_channels=3, scale_factor=2):
        super().__init__()
        self.stn = STN(in_channels=in_channels)

        # Feature extraction for upsampling
        self.feature_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )

        # Upscale block using PixelShuffle
        self.upscale = nn.Sequential(
            nn.Conv2d(64, 64 * (scale_factor ** 2), kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.PReLU(),
        )

    def forward(self, x):
        """
        Args:
            x: [B*T, C, H, W]
        Returns:
            upsampled: [B*T, C, H*scale, W*scale]
        """
        x = self.stn(x)
        feat = self.feature_conv(x)
        upsampled = self.upscale(feat)
        return upsampled
