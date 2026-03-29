"""
Comparative training: EDSR-16 (×2) vs Real-ESRGAN (×4) for LPR.

Trains two models side-by-side with identical recognition heads so that
the SR module is the only variable.

Usage:
    uv run train_compare.py

Outputs (git-ignored):
    best_model_edsr.pth
    best_model_realesrgan.pth
    val_tracks.json          ← shared with train.py (80/20 split)
    train_results.json        ← per-epoch metrics for both models
"""

from __future__ import annotations

import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

try:
    from .config import Config
    from .dataset import EndToEndDataset
    from .models.crnn import EndToEndLPR, EDSR
    from .models.realesrgan import RealESRGANGenerator
    from .utils import seed_everything, decode_predictions
except ImportError:
    from config import Config
    from dataset import EndToEndDataset
    from models.crnn import EndToEndLPR, EDSR
    from models.realesrgan import RealESRGANGenerator
    from utils import seed_everything, decode_predictions


# ---------------------------------------------------------------------------
# Hybrid models: SR module + shared recognition head
# ---------------------------------------------------------------------------

class EDSR_LPR(nn.Module):
    """
    EDSR-16 (×2) + LPR recognition head.
    SR output: 32×128 → 64×256   →  Swin: [2,8] → 32 CTC timesteps.
    """

    def __init__(self, num_classes: int, d_model: int = 512):
        super().__init__()
        self.upscale_factor = 2

        # ── SR: EDSR-16 (pretrained-compatible, 21.8 M params) ──
        from models.crnn import EDSR
        self.sr_module = EDSR()

        # ── Recognition (same as EndToEndLPR) ──
        self._build_recognition(num_classes, d_model, sr_spatial=(2, 8))

    def _build_recognition(self, num_classes, d_model, sr_spatial):
        from torchvision.models import swin_t, Swin_T_Weights
        import torch.nn.functional as F

        _swin = swin_t(weights=Swin_T_Weights.DEFAULT)
        self.backbone = _swin.features

        h_tokens, w_tokens = sr_spatial
        # Width upscale ×(T_target/T_current) to reach 32 CTC timesteps
        # 32 / w_tokens
        w_up = max(1, round(32 / w_tokens))

        self.conv_proj = nn.Sequential(
            nn.Upsample(scale_factor=(1, w_up), mode='bilinear', align_corners=False),
            nn.Conv2d(768, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None)),
        )

        self.pos_encoder = self._make_pos_encoder(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=8)
        self.fc = nn.Linear(d_model, num_classes)

        # Store target T for CTC input_lengths
        self.T = max(1, w_tokens * w_up)

    @staticmethod
    def _make_pos_encoder(d_model, max_len=5000):
        import math
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        class PosEnc(nn.Module):
            def __init__(self, pe_tensor):
                super().__init__()
                self.register_buffer('pe', pe_tensor)
            def forward(self, x):
                return x + self.pe[:, :x.size(1)]
        return PosEnc(pe)

    def forward(self, x):
        sr_img = self.sr_module(x)                     # [B, 3, 64, 256] or [B,3,128,512]
        features = self.backbone(sr_img)                # [B, C, H', W']
        features = features.permute(0, 3, 1, 2)        # [B, 768, H', W']

        features = self.conv_proj(features)             # [B, 512, 1, T]
        T = features.size(3)
        seq = features.squeeze(2).permute(0, 2, 1)   # [B, T, 512]

        seq = self.pos_encoder(seq)
        out = self.transformer(seq)
        out = self.fc(out)

        return sr_img, out.log_softmax(2)


class RealESRGAN_LPR(nn.Module):
    """
    Real-ESRGAN (×4) + LPR recognition head.
    SR output: 32×128 → 128×512  →  Swin: [4,16] → 64 CTC timesteps.
    """

    def __init__(self, num_classes: int, d_model: int = 512):
        super().__init__()
        self.upscale_factor = 4

        # ── SR: Real-ESRGAN ×4 (1.4 M params) ──
        from models.realesrgan import RealESRGANGenerator
        self.sr_module = RealESRGANGenerator(upscale_factor=4)

        # ── Recognition ──
        self._build_recognition(num_classes, d_model)

    def _build_recognition(self, num_classes, d_model):
        from torchvision.models import swin_t, Swin_T_Weights

        _swin = swin_t(weights=Swin_T_Weights.DEFAULT)
        self.backbone = _swin.features

        # Swin output for 128×512: [B, 4, 16, 768] → 16×4=64 spatial tokens
        # Upsample width ×4 to get 64 CTC timesteps
        self.conv_proj = nn.Sequential(
            nn.Upsample(scale_factor=(1, 4), mode='bilinear', align_corners=False),
            nn.Conv2d(768, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None)),
        )

        self.pos_encoder = self._make_pos_encoder(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=8)
        self.fc = nn.Linear(d_model, num_classes)
        self.T = 64  # CTC timesteps

    @staticmethod
    def _make_pos_encoder(d_model, max_len=5000):
        import math
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        class PosEnc(nn.Module):
            def __init__(self, pe_tensor):
                super().__init__()
                self.register_buffer('pe', pe_tensor)
            def forward(self, x):
                return x + self.pe[:, :x.size(1)]
        return PosEnc(pe)

    def forward(self, x):
        sr_img = self.sr_module(x)                     # [B, 3, 128, 512]
        features = self.backbone(sr_img)                # [B, 4, 16, 768]
        features = features.permute(0, 3, 1, 2)        # [B, 768, 4, 16]

        features = self.conv_proj(features)             # [B, 512, 1, 64]
        seq = features.squeeze(2).permute(0, 2, 1)   # [B, 64, 512]

        seq = self.pos_encoder(seq)
        out = self.transformer(seq)
        out = self.fc(out)

        return sr_img, out.log_softmax(2)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

MODELS = {
    'EDSR-16':  (EDSR_LPR,      'best_model_edsr.pth',     (Config.IMG_HEIGHT * 2,  Config.IMG_WIDTH * 2)),
    'RealESRGAN': (RealESRGAN_LPR, 'best_model_realesrgan.pth', (Config.IMG_HEIGHT * 4,  Config.IMG_WIDTH * 4)),
}


def train_model(model_name, model_cls, ckpt_path, hr_h, hr_w):
    """Train a single model and return per-epoch metrics."""
    seed_everything(Config.SEED)
    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"{'='*60}")

    # Datasets — use shared val_tracks.json split
    train_ds = EndToEndDataset(Config.DATA_ROOT, mode='train', split_ratio=0.8)
    val_ds   = EndToEndDataset(Config.DATA_ROOT, mode='val',   split_ratio=0.8)

    if len(train_ds) == 0:
        print(f"  Dataset empty — skipping {model_name}")
        return None

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=EndToEndDataset.collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=EndToEndDataset.collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    ) if len(val_ds) > 0 else None

    # Model
    device = Config.DEVICE
    model = model_cls(num_classes=Config.NUM_CLASSES).to(device)

    # Count params
    sr_params  = sum(p.numel() for p in model.sr_module.parameters()) / 1e6
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  SR params: {sr_params:.1f}M  |  Total params: {total_params:.1f}M")
    print(f"  SR output:  {hr_h}×{hr_w}  |  CTC T={model.T}")

    criterion_ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    criterion_sr  = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=Config.LEARNING_RATE,
        steps_per_epoch=len(train_loader), epochs=Config.EPOCHS,
        pct_start=0.3, div_factor=25.0, final_div_factor=1000.0,
        anneal_strategy='cos',
    )
    scaler = GradScaler()

    best_acc = 0.0
    epochs_data = []
    T = model.T

    for epoch in range(Config.EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_loss_ctc = 0.0
        epoch_loss_sr  = 0.0

        pbar = tqdm(train_loader, desc=f'[{model_name}] Ep {epoch+1}/{Config.EPOCHS}')
        for lr_images, hr_images, targets, target_lengths, _, _ in pbar:
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)
            targets   = targets.to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda'):
                sr_imgs, preds = model(lr_images)

                preds_permuted = preds.permute(1, 0, 2)   # [T, B, C]
                input_lengths  = torch.full((lr_images.size(0),), T, dtype=torch.long)

                loss_ctc = criterion_ctc(preds_permuted, targets, input_lengths, target_lengths)
                loss_sr  = criterion_sr(sr_imgs, hr_images)
                loss     = loss_ctc + 1.0 * loss_sr

            scaler_scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scaler_scale_before:
                scheduler.step()

            epoch_loss     += loss.item()
            epoch_loss_ctc += loss_ctc.item()
            epoch_loss_sr  += loss_sr.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'ctc':  f'{loss_ctc.item():.3f}',
                'sr':   f'{loss_sr.item():.3f}',
                'lr':   f'{scheduler.get_last_lr()[0]:.1e}',
            })

        avg_train_loss    = epoch_loss     / len(train_loader)
        avg_train_loss_ctc = epoch_loss_ctc / len(train_loader)
        avg_train_loss_sr  = epoch_loss_sr  / len(train_loader)

        # Validation
        val_acc = 0.0
        if val_loader:
            model.eval()
            total_correct = 0
            total_samples  = 0

            with torch.no_grad():
                for lr_images, hr_images, targets, target_lengths, labels_text, _ in val_loader:
                    lr_images = lr_images.to(device)
                    hr_images  = hr_images.to(device)
                    targets    = targets.to(device)

                    sr_imgs, preds = model(lr_images)

                    _ = criterion_ctc(
                        preds.permute(1, 0, 2), targets,
                        torch.full((lr_images.size(0),), T, dtype=torch.long),
                        target_lengths,
                    )

                    decoded = decode_predictions(torch.argmax(preds, dim=2), Config.IDX2CHAR)
                    for gt, pred in zip(labels_text, decoded):
                        if pred == gt:
                            total_correct += 1
                    total_samples += len(labels_text)

            val_acc = (total_correct / total_samples * 100) if total_samples > 0 else 0.0

        epoch_data = {
            'epoch': epoch + 1,
            'train_loss':     round(avg_train_loss,     4),
            'train_loss_ctc': round(avg_train_loss_ctc, 4),
            'train_loss_sr':  round(avg_train_loss_sr,  4),
            'val_acc':        round(val_acc, 2),
        }
        epochs_data.append(epoch_data)

        print(
            f"  [{model_name}] Ep {epoch+1:02d}/{Config.EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} (CTC {avg_train_loss_ctc:.3f} SR {avg_train_loss_sr:.3f}) | "
            f"Val Acc: {val_acc:.2f}%"
        )

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"    ★ Best model saved ({val_acc:.2f}%)")

    print(f"\n  {model_name} done — best Val Acc: {best_acc:.2f}%")
    return epochs_data, best_acc


def main():
    seed_everything(Config.SEED)
    print(f"\n{'#'*60}")
    print(f"#  Comparative Training: EDSR-16 (×2) vs Real-ESRGAN (×4)")
    print(f"#  Device: {Config.DEVICE}  |  Epochs: {Config.EPOCHS}")
    print(f"{'#'*60}")

    if not os.path.exists(Config.DATA_ROOT):
        print(f"  DATA_ROOT not found: {Config.DATA_ROOT}")
        return

    all_results = {}
    start_time = time.time()

    for name, (model_cls, ckpt_path, (hr_h, hr_w)) in MODELS.items():
        epochs_data, best_acc = train_model(name, model_cls, ckpt_path, hr_h, hr_w)
        if epochs_data is not None:
            all_results[name] = {
                'epochs': epochs_data,
                'best_val_acc': best_acc,
                'ckpt': ckpt_path,
                'hr_size': f'{hr_h}×{hr_w}',
            }

    elapsed = time.time() - start_time

    # Summary
    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"  Total time: {elapsed/60:.1f} min")
    print(f"\n  {'Model':<12} {'SR size':<12} {'Best Val Acc':>12} {'SR params':>10}")
    print(f"  {'-'*50}")

    summary = {}
    for name, data in all_results.items():
        sr = data['hr_size']
        acc = data['best_val_acc']
        summary[name] = {'best_val_acc': acc, 'hr_size': sr}
        # Fetch SR param count
        model = MODELS[name][0](Config.NUM_CLASSES)
        sr_p = sum(p.numel() for p in model.sr_module.parameters()) / 1e6
        print(f"  {name:<12} {sr:<12} {acc:>11.2f}% {sr_p:>9.1f}M")

    # Save results
    results_path = 'train_results.json'
    with open(results_path, 'w') as f:
        json.dump({'summary': summary, 'elapsed_minutes': round(elapsed/60, 1)}, f, indent=2)
    print(f"\n  Results saved to: {results_path}")


if __name__ == '__main__':
    main()
