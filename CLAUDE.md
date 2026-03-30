# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Multi-Frame License Plate Recognition (LPR)** — recognizes plate text from video tracks using a ConvNeXt + Transformer CRNN architecture with CTC loss. Handles Brazilian/Mercosur-format plates (3 letters + 4 digits, e.g. `AVL5215`). Supports mixed-resolution input: 50/50 LR frames and degraded HR frames during training.

## Commands

```bash
# Download dataset (train + test)
uv run python download_data.py

# Install / sync dependencies
uv sync

# Train (saves best_model.pth)
uv run python train.py

# Evaluate on held-out test set (loads best_model2.pth)
uv run python test.py

# Quick architecture sanity check
uv run python verify_haf.py

# Patch notebook with latest config/architecture
uv run python update_nb.py
```

## Architecture

```
Input [B, T=5, 3, 64, 256]
    ↓
STN (geometric correction)  →  [B*T, 3, 64, 256]
    ↓
RealESRGAN ×2 (2× PixelShuffle each)  →  [B*T, 3, 256, 1024]
    ↓
ConvNeXt Tiny Backbone (pretrained)  →  [B*T, 768, 8, 32]
    ↓
ConvProj (768→512, H:8→1)  →  [B*T, 512, 1, 32]
    ↓
HybridAttentionFusion  →  [B, 512, 1, 32]
    ↓
AdaptiveAvgPool + Permute  →  [B, 32, 512]
    ↓
PositionalEncoding (sinusoidal) + TransformerEncoder (6 layers, d=512, 8 heads)
    ↓
FC + LogSoftmax  →  [B, 32, 38]
```

**HybridAttentionFusion** combines three mechanisms:
1. **Temporal Gating** — per-frame quality weight via pooled features → sigmoid
2. **Ref-Aware Spatial Attention** — each frame concatenated with center frame, scored via Conv2d, softmax-weighted sum
3. **Channel Attention** — SE-style post-fusion channel refinement

**STN** — Spatial Transformer Network. Predicts affine θ per-frame via a localization CNN; applies `F.grid_sample` for geometric correction (rotation, scale, translation).

**RealESRGAN** — Lightweight Real-ESRGAN-style upsampler. Each block: 2× RRDB (Residual-in-Residual Dense Block) + PixelShuffle upscale + skip connection from upsampled input. Two stacked blocks give 4× total super-resolution.

## Data Format

```
data/train/  (data/train/Scenario-A/{Brazilian,Mercosur}/track_XXXXX/)
    ├── track_XXXXX/
    │   ├── annotations.json   # {"plate_text": "AVL5215", "plate_layout": "Brazilian", "corners": {...}}
    │   ├── lr-001.png … lr-005.png  # Low-res (~32×16 px)
    │   └── hr-001.png … hr-005.png  # High-res (~60×30 px)

data/test/  (blind test — images only, no labels)
    └── track_XXXXX/
        └── lr-001.jpg … lr-005.jpg
```

## Train / Val / Test Split

- **Total**: 20,000 annotated tracks in `data/train/`
- **Test**: 2,000 held-out tracks (from `notebook/test_tracks.json`), excluded from train/val
- **Train**: 14,400 (80% of remaining)
- **Val**: 3,600 (20% of remaining)
- `val_tracks.json` — persisted train/val split (seed=42). Regenerate by deleting the file.

The `AdvancedMultiFrameDataset` automatically excludes test tracks. `TestDataset` reads test tracks from `data/train/` using `notebook/test_tracks.json`.

## Key Training Details

- **STN + RealESRGAN**: enabled by `Config.USE_STN` / `Config.USE_SR` (default: True)
- **Backbone**: ConvNeXt Tiny frozen for epochs 0–4, then 30% random freeze probability per epoch
- **Mixup**: 15% chance, beta(1,1), two CTC losses weighted by λ
- **Scheduler**: OneCycleLR — 30% warmup, cosine annealing, max_lr=3e-4
- **Mixed precision**: AMP GradScaler on CUDA, gradient clipping max_norm=1.0
- **Augmentation**: affine/elastic/perspective + brightness + blur/noise + coarse dropout (albumentations)
- **HR degradation**: blur + noise + JPEG compression + random downscaling
- **CTC decoding**: greedy (beam_width=1) during training/val; beam search (beam_width=5) + plate format filter only in `test.py`

## Model File Naming — Important

- `train.py` saves to `best_model.pth`
- `test.py` loads from `best_model2.pth` — these are **separate files**
- Both use `strict=False` loading to tolerate minor architecture changes

## Key Config Options (`config.py`)

| Variable | Default | Description |
|---|---|---|
| `USE_STN` | True | Enable STN geometric correction |
| `USE_SR` | True | Enable RealESRGAN super-resolution |
| `SR_SCALE` | 2 | Upscale factor per RRDB block |
| `BATCH_SIZE` | 64 | Reduce on CPU / low-VRAM GPUs |
| `NUM_FRAMES` | 8 | Intent (dataset caps at 5) |

## Known Discrepancies

| Issue | Detail |
|---|---|
| Frame count | `Config.NUM_FRAMES=8` but dataset always pads/truncates to **5 frames** |
| Dead code | `ultimate_sync.py` has hardcoded Windows paths — not used |
| Stale docs | `README.md` references Swin Transformer; actual backbone is ConvNeXt Tiny |

## Key Utilities (`utils.py`)

- `decode_predictions(log_probs, idx2char, beam_width, use_format_filter)` — CTC decoding (greedy or beam search)
- `vietnam_plate_score(text)` — scores plate format validity (used by format filter in beam search)
- `seed_everything(seed)` — sets all random seeds for reproducibility
