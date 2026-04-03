# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A PyTorch research project for the **ICPR 2026 competition**: an end-to-end pipeline that takes **low-resolution (32×128) license plate images**, super-resolves them to high-resolution (64×256), and recognizes the plate text using a Swin Transformer + Transformer Encoder with CTC loss.

> **Critical**: `DATA_ROOT = "data/train"` is hardcoded in `config.py`. Training silently skips if no tracks are found — set it to the ICPC dataset root before running.

## Tech Stack

| | |
|---|---|
| Language | Python ≥ **3.13** (enforced via `.python-version` and `pyproject.toml`) |
| ML Framework | PyTorch 2.x + torchvision |
| Package Manager | **uv** only — use `uv sync` and `uv run <script>`. Do not use pip. |
| Dependencies | `pyproject.toml` + `uv.lock` (lock file is the source of truth for versions) |
| Data Augmentation | albumentations ≥ 1.3 |
| Image I/O | OpenCV, Pillow |

## Commands

```bash
# Sync dependencies (run once after cloning or pulling)
uv sync

# Train the primary model: EDSR-16 super-resolution + Swin-T + CTC
# Writes: best_model.pth, val_tracks.json
uv run train.py

# Comparative study: train EDSR-16 (×2) and Real-ESRGAN (×4) side-by-side
# Writes: best_model_edsr.pth, best_model_realesrgan.pth, train_results.json
uv run train_compare.py

# Evaluate on test set (reads best_model.pth)
# Writes: predictions.json, test_results.json
uv run test.py
```

No test framework (pytest/unittest) is used. `test.py` is an evaluation/inference script, not a unit test.

## Architecture

```
Input: LR image [B, 3, 32, 128]
  │
  ▼
┌──────────────────────────┐
│  SR Module (configurable) │  ← chosen by train script
│  EDSR-16: 16 ResBlocks, 256 features, PixelShuffle ×2 → [B,3,64,256]
│  Real-ESRGAN: 6 RRDB blocks, PixelShuffle ×4 → [B,3,64,256]
└──────────────┬───────────┘
               │ [B, 3, 64, 256] (also used for MSELoss)
               ▼
┌──────────────────────────┐
│  Swin-Tiny Backbone      │  ← frozen, pretrained (torchvision.models.swin_t)
│  → [B, 768, 2, 8]        │
└──────────────┬───────────┘
               │ permute → ConvProj (768→512, H→1) → AdaptiveAvgPool
               ▼
┌──────────────────────────┐
│  TransformerEncoder       │  ← 8 layers, 8 heads, d_model=512
│  + PositionalEncoding     │
│  → [B, 32, 512]           │
└──────────────┬───────────┘
               ▼
┌──────────────────────────┐
│  FC → LogSoftmax          │  → [B, 32, NUM_CLASSES]
│  CTC Loss (blank=0)       │
└──────────────────────────┘
```

**Multi-task loss**: `total_loss = CTC_Loss + λ_SR * MSE_Loss(sr_img, hr_img)` where `λ_SR = 1.0` in `train.py`.

## Project Structure

```
ICPR-2026/
├── config.py              ← single Config class (LR/HR sizes, CHARS, hyperparameters)
├── dataset.py             ← EndToEndDataset (train/val/test splits, LR/HR/label loading)
├── train.py               ← train with EDSR-16 only
├── train_compare.py       ← comparative training: EDSR-16 vs Real-ESRGAN
├── test.py                ← evaluation and inference
├── transforms.py          ← augmentation pipelines (train augment, val, degradation)
├── utils.py               ← seed_everything(), decode_predictions()
├── models/
│   ├── crnn.py            ← EndToEndLPR, EDSR, EDSRResBlock, PositionalEncoding
│   └── realesrgan.py       ← RealESRGANGenerator, RealESRGANGeneratorLite, RRDB
├── notebook/
│   └── baseline.ipynb     ← older experimental MultiFrameCRNN (T=5, RefAwareFusion)
│                             ⚠ superseded by main pipeline; do not use as reference
├── val_tracks.json        ← auto-generated 80/20 train/val split (git-ignored)
├── best_model.pth         ← best checkpoint from train.py (git-ignored)
└── data/                  ← not in repo; set DATA_ROOT in config.py
    └── train/
        └── Scenario-A/
            ├── Brazilian/track_XXXXX/
            │   ├── annotations.json   # {"plate_text": "ABC123"} or [{"plate_text": "..."}]
            │   ├── lr-00.png ...       # low-res frames (32×128)
            │   └── hr-00.png ...       # high-res frames (64×256), optional
            └── test/track_XXXXX/       # no annotations required
```

## Configuration (config.py)

| Parameter | Value | Notes |
|---|---|---|
| `IMG_HEIGHT/WIDTH` | 32 / 128 | LR input |
| `HR_IMG_HEIGHT/WIDTH` | 64 / 256 | SR output |
| `CHARS` | `0-9A-Z-` (38) | Character set; `-` is the plate separator |
| `NUM_CLASSES` | 39 | `len(CHARS) + 1` (CTC blank at index 0) |
| `BATCH_SIZE` | 128 | |
| `LEARNING_RATE` | 5e-4 | |
| `EPOCHS` | 80 | |
| `SEED` | 42 | |

Training: `AdamW(lr=5e-4, weight_decay=1e-4)`, `OneCycleLR`, AMP (`torch.amp.autocast('cuda')`), gradient clipping.

## Important Conventions

- **CTC decoding**: Greedy decode (argmax → collapse repeats → remove blanks). Index `0` is the CTC blank.
- **Reproducibility**: Call `seed_everything(Config.SEED)` before any stochastic operation.
- **Dual-import pattern**: All scripts use `from .module` / `from module` to support both `uv run` and direct `python`.
- **No multi-frame support in current model**: `EndToEndLPR` processes single images. The old `MultiFrameCRNN` in `notebook/baseline.ipynb` (T=5, `RefAwareFusion`) is superseded.

## Common Customizations

```python
# Add an augmentation — edit transforms.py → get_train_transforms()
# Change character set — update Config.CHARS in config.py (requires full retrain)
# Switch SR model — use train_compare.py for Real-ESRGAN; in train.py swap EDSR for RealESRGANGenerator
# Adjust SR loss weight — change lambda_sr in train.py
# Switch vision backbone — replace swin_t() in models/crnn.py → EndToEndLPR.__init__
# Adjust EDSR depth — change num_blocks in EDSR.__init__ (16=EDSR-16, 32=EDSR-32)
```

## Known Limitations

- `val_tracks.json` is only created by `train.py`, not by `train_compare.py` or `test.py`.
- Split key is `os.path.basename(track_path)` — if tracks are renamed/moved the file becomes stale (delete and re-run).
- `best_model.pth` is loaded with `weights_only=True` — changing model architecture requires retraining from scratch.
- `update_nb.py` is a one-shot migration script (mobilenet_v3 → swin_t); do not re-run after model changes.
- `main.py` is a placeholder; use `train.py` / `train_compare.py` / `test.py` directly.
