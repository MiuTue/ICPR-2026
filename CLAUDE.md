# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Multi-Frame License Plate Recognition (LPR)** — recognizes plate text from video tracks using a ConvNeXt + Transformer CRNN architecture with CTC loss. Handles Brazilian/Mercosur-format plates (3 letters + 4 digits, e.g. `AVL5215`). Supports mixed-resolution input: 50/50 LR frames and degraded HR frames during training.

## Commands (Vast.ai / pip)

```bash
# Install dependencies
pip install -r requirements.txt

# Download dataset (run once)
bash run.sh download

# Train — all params via env vars
WANDB_API_KEY=your_key \
BATCH_SIZE=16 \
NUM_WORKERS=4 \
EPOCHS=80 \
  bash run.sh train

# Evaluate on held-out test set
cp best_model.pth best_model2.pth
bash run.sh test

# Quick architecture sanity check
python verify_haf.py

# Patch notebook with latest config/architecture
python update_nb.py
```

**Vast.ai environment variables:**

| Variable | Default | Description |
|---|---|---|
| `WANDB_API_KEY` | — | Required for training |
| `WANDB_PROJECT` | `icpr-2026-lpr` | W&B project name |
| `BATCH_SIZE` | `16` | Adjust for GPU memory |
| `NUM_WORKERS` | `4` | DataLoader workers |
| `EPOCHS` | `80` | Number of epochs |
| `LR` | `0.0003` | Learning rate |
| `USE_STN` | `1` | Enable STN (0/1) |
| `USE_SR` | `1` | Enable RealESRGAN (0/1) |
| `DATA_ROOT` | `data/train` | Path to train data |

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

**HybridAttentionFusion** — 3 mechanisms combined:
1. **Temporal Gating** — per-frame quality weight via pooled features → sigmoid
2. **Ref-Aware Spatial Attention** — each frame concatenated with center frame, softmax-weighted sum
3. **Channel Attention** — SE-style post-fusion channel refinement

**STN** — Spatial Transformer. Localization CNN predicts 6 affine params per frame; `F.grid_sample` applies correction.

**RealESRGAN** — 2 stacked blocks: 2× RRDB + PixelShuffle + skip connection. Total 4× super-resolution.

## Data Format

```
data/train/  (Scenario-A/{Brazilian,Mercosur}/track_XXXXX/)
    └── track_XXXXX/
        ├── annotations.json   # {"plate_text": "AVL5215", "plate_layout": ..., "corners": {...}}
        ├── lr-001.png … lr-005.png  # Low-res (~32×16 px)
        └── hr-001.png … hr-005.png  # High-res (~60×30 px)
```

## Train / Val / Test Split

| Set | Size | Source |
|---|---|---|
| Train | 14,400 | 80% of non-test tracks |
| Val | 3,600 | 20% of non-test tracks |
| Test | 2,000 | `notebook/test_tracks.json` |

- `AdvancedMultiFrameDataset` auto-excludes test tracks from train/val
- `TestDataset` reads test tracks from `data/train/` using `TEST_TRACKS_FILE`
- `val_tracks.json` — persisted split (seed=42). Delete to regenerate.

## Key Training Details

- **Gradual unfreezing**: phase 0 (ep 0–4) freeze backbone+SR/STN → phase 1 (ep 5–9) unlock SR → phase 2 (ep 10+) stochastic backbone freeze
- **Mixup**: 15% chance, beta(1,1)
- **Scheduler**: OneCycleLR — 30% warmup, cosine annealing
- **Mixed precision**: AMP GradScaler on CUDA, grad clip max_norm=1.0
- **Augmentation**: affine/elastic/perspective + brightness + blur/noise + coarse dropout
- **CTC decoding**: greedy (beam_width=1) train/val; beam search (beam_width=5) + format filter in `test.py`

## Model File Naming

- `train.py` saves → `best_model.pth`
- `test.py` loads → `best_model2.pth` (**separate files**)
- Both use `strict=False` loading

## Weights & Biases Logging

Project: `icpr-2026-lpr`. Run URL printed at start of training.

| Frequency | Metrics |
|---|---|
| Every 10 steps | `step/loss`, `step/lr`, `step/grad_norm`, `step/phase` |
| Every 50 steps | Prediction table (frame thumbnail + GT vs predicted text) |
| Every epoch | `train_loss`, `val_loss`, `val_acc`, `val_char_acc`, duration, LR |
| On improvement | `best_model.pth` + W&B Artifact upload |

`wandb.watch(model)` logs gradients periodically. Config (model, data, hyperparams) auto-logged.

## Key Utilities (`utils.py`)

- `decode_predictions(log_probs, idx2char, beam_width, use_format_filter)` — CTC greedy or beam search
- `vietnam_plate_score(text)` — plate format validity scorer (used by beam search filter)
- `seed_everything(seed)` — reproducibility across all random libs

## Known Discrepancies

| Issue | Detail |
|---|---|
| Frame count | `Config.NUM_FRAMES=8` but dataset always caps at **5 frames** |
| Dead code | `ultimate_sync.py` has hardcoded Windows paths — unused |
| Stale docs | `README.md` references Swin Transformer; actual backbone is ConvNeXt Tiny |
