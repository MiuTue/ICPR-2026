"""
Module 2: Multi-Frame License Plate Recognition CRNN.

Pipeline:
    Input [B, T=5, 3, 64, 256]
        │
        │  Module 1: LightSR (được gọi bên ngoài hoặc trong forward)
        ▼
    LR → SR  →  [B*T, 3, 256, 1024]
        │
        ▼  Module 2: Recognition
    ConvNeXt Tiny Backbone  →  [B*T, 768, 8, 32]
        │
        ▼
    ConvProj (768→256)  →  [B*T, 256, 8, 32]
        │
        ▼
    Select center frame  →  [B, 256, 8, 32]
        │
        ▼
    AdaptiveAvgPool (W:32→1) + Permute  →  [B, 8, 256]
        │
        ▼
    Bidirectional LSTM (1 layer, 256 hidden)  →  [B, 8, 512]
        │
        ▼
    FC + LogSoftmax  →  [B, 8, num_classes]

Lý do:
    - STN: bỏ (biển số đã tương đối thẳng)
    - Multi-frame fusion: chọn center frame (plate static, các frame giống nhau)
    - Transformer: thay bằng BiLSTM nhẹ hơn, đủ cho sequence 7 ký tự
    - ConvProj stride: bỏ stride trên H, giữ nguyên spatial info
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

try:
    from .light_sr import LightSR
except ImportError:
    from light_sr import LightSR


class MultiFrameCRNN(nn.Module):
    """
    2-Module LPR Pipeline:
        Module 1: LightSR (upscale LR → HR)
        Module 2: ConvNeXt + BiLSTM + CTC
    """
    def __init__(self, num_classes, d_model=256, use_sr=True):
        super().__init__()
        self.use_sr = use_sr

        # ── Module 1: Super-Resolution ──────────────────────────────────────
        if use_sr:
            self.sr = LightSR(
                in_channels=3,
                hidden_channels=64,
                num_res_blocks=2
            )

        # ── Module 2: Recognition ───────────────────────────────────────────
        # Backbone: ConvNeXt Tiny (pretrained)
        convnext = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        self.backbone = convnext.features  # Output: 768 channels

        # Projection: 768 → d_model, giảm H từ 8 → 1
        self.conv_proj = nn.Sequential(
            nn.Conv2d(768, d_model, kernel_size=3, stride=(1, 1), padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(True),
        )

        # BiLSTM: sequence modeling cho plate characters
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Output: 2× hidden = d_model*2 (vì bidirectional)
        self.fc = nn.Linear(d_model * 2, num_classes)

    def freeze_sr(self, freeze=True):
        """Freeze or unfreeze the SR module."""
        if self.use_sr:
            for param in self.sr.parameters():
                param.requires_grad = not freeze

    def forward(self, x, use_sr=None, return_features=False):
        """
        Args:
            x: [B, T=5, 3, 64, 256] — 5 LR frames
            use_sr: override SR flag (None = use self.use_sr)
            return_features: if True, return intermediate features for debugging
        Returns:
            out: [B, T_out=32, num_classes] — CTC log-probs
        """
        if use_sr is None:
            use_sr = self.use_sr

        b, t, c, h, w = x.size()

        # ── Module 1: SR ──────────────────────────────────────────────────────
        if use_sr:
            x = x.view(b * t, c, h, w)   # [B*T, 3, 64, 256]
            x = self.sr(x)               # [B*T, 3, 256, 1024] (4×)
            _, _, h_sr, w_sr = x.shape
            x = x.view(b, t, c, h_sr, w_sr)  # [B, T, 3, 256, 1024]
        else:
            # Không SR: reshape luôn
            x = x.view(b, t, c, h, w)

        # ── Module 2: Recognition ───────────────────────────────────────────
        # Merge Batch & Time
        x = x.view(b * t, c, x.size(3), x.size(4))  # [B*T, 3, H, W]

        # ConvNeXt backbone
        features = self.backbone(x)      # [B*T, 768, H/32, W/32]

        # Conv projection: 768 → 256
        features = self.conv_proj(features)  # [B*T, 256, 8, 32]

        # Chọn center frame (plate static → các frame giống nhau)
        center_idx = t // 2
        features = features.view(b, t, *features.shape[1:])  # [B, T, 256, 8, 32]
        fused = features[:, center_idx]                        # [B, 256, 8, 32]

        # Pool to fixed [H=8, W=1] → sequence length always 8
        # With SR:  H=8, W=32 → pool  H→8, W→1
        # Without SR: H=2, W=8  → pool  H→8, W→1
        seq = F.adaptive_avg_pool2d(fused, (8, 1))           # [B, 256, 8, 1]
        seq = seq.squeeze(3).permute(0, 2, 1)                # [B, 8, 256]

        # BiLSTM
        lstm_out, _ = self.lstm(seq)     # [B, 8, 512]

        # FC + log_softmax
        out = self.fc(lstm_out)           # [B, 8, num_classes]
        out = F.log_softmax(out, dim=2)  # CTC expects [B, T, C]

        if return_features:
            return out, features

        return out
