# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Multi-Frame License Plate Recognition (LPR)** — 2-module pipeline:
- **Module 1 (LightSR):** Upscales LR frames (64×256) → HR frames (256×1024), 4× scale
- **Module 2 (CRNN):** ConvNeXt Tiny + BiLSTM + CTC — recognizes plate text from HR frames

Supports Brazilian/Mercosur-format plates (3 letters + 4 digits, e.g. `AVL5215`).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Download dataset
bash run.sh download

# Train
WANDB_API_KEY=your_key BATCH_SIZE=16 NUM_WORKERS=4 EPOCHS=80 bash run.sh train

# Evaluate on held-out test set
cp best_model.pth best_model2.pth
bash run.sh test

# Quick architecture sanity check
python verify_haf.py
```

**Environment variables:**

| Variable | Default | Description |
|---|---|---|
| `WANDB_API_KEY` | — | Required for training |
| `WANDB_PROJECT` | `icpr-2026-lpr` | W&B project name |
| `BATCH_SIZE` | `4` | Micro-batch |
| `NUM_WORKERS` | `4` | DataLoader workers |
| `EPOCHS` | `80` | Number of epochs |
| `LR` | `0.0003` | Learning rate |
| `USE_SR` | `1` | Enable LightSR (0/1) |
| `DATA_ROOT` | `data/train` | Path to train data |

## Architecture

### Module 1: LightSR (Super-Resolution)

```
Input [B*T, 3, 64, 256]
    → Shallow feature (3→64) + ReLU
    → 2× ResidualBlock
    → Fusion conv
    → Up1: PixelShuffle 2× → [B*T, 3, 128, 512]
    → Up2: PixelShuffle 2× → [B*T, 3, 256, 1024]
    + Bilinear 4× skip connection
Output [B*T, 3, 256, 1024]
```

### Module 2: Recognition CRNN

```
Input [B, T=5, 3, 64, 256]
    │
    │  Module 1 (LightSR): 4× upscale
    ▼
SR frames [B*T, 3, 256, 1024]
    │
    ▼  Module 2
ConvNeXt Tiny (pretrained) → [B*T, 768, 8, 32]
    ↓
ConvProj (768→256) → [B*T, 256, H', W']
    ↓
Select center frame → [B, 256, H', W']
    ↓
AdaptiveAvgPool (H'→8, W'→1) + Permute → [B, 8, 256]
    ↓
BiLSTM (1 layer, 256 hidden) → [B, 8, 512]
    ↓
FC + LogSoftmax → [B, 8, 37]
```

**Design decisions:**
- **No STN:** License plate images are already relatively straight
- **No multi-frame fusion:** All frames in a track are identical (plate is static)
- **BiLSTM instead of Transformer:** 7 characters is too short for Transformer; LSTM is lighter and sufficient
- **Center frame selection:** Simple and effective since all frames are the same

## Data Format

```
data/train/  (Scenario-A/{Brazilian,Mercosur}/track_XXXXX/)
    └── track_XXXXX/
        ├── annotations.json   # {"plate_text": "AVL5215"}
        ├── lr-001.png … lr-005.png  # Low-res (~50×25 px)
        └── hr-001.png … hr-005.png  # High-res (~60×30 px, optional in train)

Test set: only LR frames (5 per track) → Module 1 generates HR → Module 2 recognizes
```

## Train / Val / Test Split

| Set | Size | Source |
|---|---|---|
| Train | 80% | Non-test tracks |
| Val | 20% | Non-test tracks |
| Test | 2000 tracks | `notebook/test_tracks.json` |

- Test tracks excluded from train/val
- `val_tracks.json` — persisted split (seed=42). Delete to regenerate.

## Key Training Details

- **Gradual unfreezing:** phase 0 (ep 0–4) freeze SR+backbone → phase 1 (ep 5–9) unlock SR → phase 2 (ep 10+) stochastic backbone freeze
- **Mixup:** 15% chance, beta(1,1)
- **Scheduler:** OneCycleLR — 30% warmup, cosine annealing
- **Mixed precision:** AMP GradScaler on CUDA, grad clip max_norm=1.0
- **Augmentation:** affine/elastic/perspective + brightness + blur/noise + coarse dropout
- **CTC decoding:** greedy (beam_width=1) train/val; beam search (beam_width=5) + Brazilian plate format filter in test

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

## Key Utilities (`utils.py`)

- `decode_predictions(log_probs, idx2char, beam_width, use_format_filter)` — CTC greedy or beam search
- `brazil_plate_score(text)` — Brazilian plate format validity scorer (used by beam search filter)
- `seed_everything(seed)` — reproducibility

## Character Set

Brazilian/Mercosur format: `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ` (36 classes + 1 CTC blank = 37 total)
