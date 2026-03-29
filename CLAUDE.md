# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **PyTorch research project** for the **ICPR 2026 competition**: an end-to-end pipeline that takes **low-resolution (32×128) license plate images**, super-resolves them to high-resolution (64×256), and recognizes the plate text using a Swin Transformer + Transformer Encoder architecture with CTC loss.

> **IMPORTANT**: `DATA_ROOT = "data/train"` is hardcoded in `config.py`. Training silently skips if no tracks are found. Point it to the ICPC dataset root before running `train.py`.

---

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python ≥ 3.13 |
| ML Framework | PyTorch 2.x + torchvision |
| Data Augmentation | albumentations ≥ 1.3 |
| Image I/O | OpenCV, Pillow |
| Environment Manager | `uv` (use `uv run <script>`) |
| Package Spec | `pyproject.toml` + `requirements.txt` |

---

## Project Structure

```
ICPR-2026/
├── CLAUDE.md              ← this file
├── README.md              ← project overview (Swin + SR pipeline, commands)
├── GEMINI.md              ← same project doc with architecture details
├── config.py              ← single Config class with all hyperparameters
├── dataset.py             ← EndToEndDataset (train/val/test splits)
├── train.py               ← multi-task training script (EDSR-16 only)
├── train_compare.py       ← comparative training: EDSR-16 (×2) vs Real-ESRGAN (×4)
├── test.py                ← evaluation & inference script
├── transforms.py          ← augmentation pipelines (train/val/degradation)
├── utils.py               ← seed_everything(), decode_predictions()
├── update_nb.py           ← one-shot notebook migration script (mobilenet→swin)
├── models/
│   ├── crnn.py            ← EndToEndLPR, EDSR, EDSRResBlock, PositionalEncoding
│   └── realesrgan.py       ← RealESRGANGenerator, RealESRGANGeneratorLite, RRDB
├── data/
│   ├── train/
│   │   ├── Scenario-A/    ← Brazilian/, Mercosur/ (country subsets)
│   │   │   └── track_XXXXX/
│   │   │       ├── annotations.json   # {"plate_text": "ABC123"}
│   │   │       ├── lr-00.png / lr-01.png ...  (low-res frames)
│   │   │       └── hr-00.png / hr-01.png ...  (high-res frames, optional)
│   │   └── test/
│   │       └── track_XXXXX/   (no annotations at test time)
├── notebook/
│   └── baseline.ipynb     ← experimental Jupyter notebook
├── val_tracks.json        ← auto-generated train/val split (git-ignored)
├── best_model.pth         ← best checkpoint (git-ignored, created by train.py)
├── predictions.json       ← test predictions (git-ignored, created by test.py)
├── test_results.json      ← test metrics (git-ignored, created by test.py)
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

---

## Key Commands

```bash
# Install / sync dependencies
uv sync

# Train EDSR-16 only (creates best_model.pth, val_tracks.json)
uv run train.py

# Comparative training: EDSR-16 (×2) vs Real-ESRGAN (×4)
# Creates best_model_edsr.pth, best_model_realesrgan.pth, train_results.json
uv run train_compare.py

# Evaluate on test set (reads best_model.pth, writes predictions.json + test_results.json)
# Searches data/test first, falls back to Config.DATA_ROOT if data/test is absent
uv run test.py

# One-shot notebook migration (mobilenet_v3 → swin_t, in-place)
uv run update_nb.py
```

---

## Architecture Summary

```
Input: LR image [B, 3, 32, 128]
  │
  ▼
┌─────────────────────────┐
│   EDSR-16               │  ← 16 ResBlocks (features=256, no BN, scale=0.1)
│   32×128  →  64×256     │  ← PixelShuffle ×2 upscale
│   21.8 M params          │  ← outputs SR image for MSELoss
└────────────┬────────────┘
             │ [B, 3, 64, 256]
             ▼
┌─────────────────────────┐
│   Swin-Tiny Backbone    │  ← torchvision.models.swin_t(weights=DEFAULT), frozen pretrained
│   output: [B, 2, 8, 768]│
└────────────┬────────────┘
             │ permute + ConvProj (Upsample×4w, Conv 768→512, AvgPool H→1)
             ▼
┌─────────────────────────┐
│   ConvProj → [B,512,1,32]│
│   AdaptiveAvgPool2d     │
│   → [B, 32, 512]        │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   PositionalEncoding    │
│   TransformerEncoder (8 layers, 8 heads, d_model=512)
│   → [B, 32, 512]        │
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│   FC → LogSoftmax       │  → [B, 32, NUM_CLASSES]
│   CTC blank index = 0  │
└─────────────────────────┘
```

### Multi-Task Loss
```
total_loss = CTC_Loss(preds, targets) + λ_SR * MSE_Loss(sr_img, hr_img)
```
where `λ_SR = 1.0` (hardcoded in `train.py`).

---

## Configuration (config.py)

All hyperparameters live in a single `Config` class. Key values:

| Parameter | Value | Notes |
|---|---|---|
| `IMG_HEIGHT / WIDTH` | 32 / 128 | LR input size |
| `HR_IMG_HEIGHT / WIDTH` | 64 / 256 | SR output size |
| `CHARS` | `0-9A-Z-` (38) | Character set; `-` is the plate separator |
| `NUM_CLASSES` | 39 | `len(CHARS) + 1` (CTC blank) |
| `BATCH_SIZE` | 128 | |
| `LEARNING_RATE` | 5e-4 | |
| `EPOCHS` | 80 | |
| `NUM_WORKERS` | 4 | DataLoader parallelism |
| `SEED` | 42 | Fixed for reproducibility |
| `VAL_SPLIT_FILE` | `val_tracks.json` | Auto-generated on first run |

---

## Data Format

### Directory layout per track
```
data/train/Scenario-A/Brazilian/track_00001/
├── annotations.json   # {"plate_text": "ABC123"}  ← required for train
├── lr-00.png          # low-res frames (32×128)
├── lr-01.png
├── hr-00.png          # high-res frames (64×256), optional
└── hr-01.png
```

### Annotation JSON (single-entry or list)
```json
{"plate_text": "ABC123"}
```
or
```json
[{"plate_text": "ABC123"}]
```

The dataset also reads `license_plate` and `text` as fallback keys.

### Degradation Pipeline (Training)
During training, the dataset optionally degrades HR images via `transforms.get_degradation_transforms()` to simulate LR inputs: Gaussian/Motion blur, GaussNoise, JPEG compression, and random downscaling.

---

## Training Details

- **Optimizer**: `AdamW(lr=5e-4, weight_decay=1e-4)`
- **Scheduler**: `OneCycleLR` (cosine annealing, pct_start=0.3)
- **Mixed Precision**: `torch.amp.autocast('cuda')` + `GradScaler`
- **Gradient Clipping**: via `torch.nn.utils.clip_grad_norm_`
- **Split Logic**: On first run, tracks are shuffled (seed=42) and split 80/20 → `val_tracks.json` is written and reused on subsequent runs.
- **Best Model**: Saved to `best_model.pth` whenever validation accuracy improves.

---

## Important Conventions

- **CTC Decoding**: Greedy decode (argmax + collapse repeats + remove blanks). Index `0` is the CTC blank.
- **Reproducibility**: Always call `seed_everything(Config.SEED)` before any stochastic operation.
- **Module Imports**: All Python files use a dual-import pattern (`from .module` / `from module`) to support both `uv run` and direct `python` execution.
- **No multi-frame support in current model**: `EndToEndLPR` processes single LR images. The older `MultiFrameCRNN` in `notebook/baseline.ipynb` handled multi-frame (T=5) inputs with `RefAwareFusion`. The current pipeline supersedes it.

---

## Known Limitations / TODOs

- `val_tracks.json` is only created by `train.py`; `test.py` does not write it.
- Test set inference does not require `annotations.json` (ground truth is optional).
- `best_model.pth` is loaded with `weights_only=True` — if the model architecture changes, retrain from scratch.
- `update_nb.py` is a one-shot migration tool (mobilenet→swin); do not re-run after model changes.
- `main.py` is a placeholder entry point; use `train.py` / `test.py` directly.
- `val_tracks.json` uses `os.path.basename(track_path)` as the split key — if tracks are renamed or moved, the split will be incorrect and the file should be deleted.

---

## Working with This Codebase

- **Add a new augmentation**: Edit `transforms.py` → `get_train_transforms()` using albumentations primitives.
- **Change character set**: Update `Config.CHARS` in `config.py` and re-run training from scratch (class indices will change).
- **Switch backbone**: Replace `swin_t(weights=Swin_T_Weights.DEFAULT)` in `models/crnn.py` → `EndToEndLPR.__init__`.
- **Adjust SR depth**: Change `num_blocks` in `EDSR.__init__` (16=EDSR-16, 32=EDSR-32; paper uses 256 features).
- **Adjust SR loss weight**: Change `lambda_sr` in `train.py`.
- **Debug a single track**: Load via `EndToEndDataset("data/train/Scenario-A/Brazilian/track_00001", mode='val')`.
