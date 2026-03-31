# Multi-Frame License Plate Recognition (LPR)

**2-Module Pipeline:**
- **Module 1 — LightSR:** Upscales LR frames (64×256) → HR frames (256×1024), 4× scale
- **Module 2 — Recognition:** ConvNeXt Tiny + BiLSTM + CTC — recognizes plate text from HR

Supports Brazilian/Mercosur-format plates (`ABC1234` — 3 letters + 4 digits).

---

## Quick Start

### 1. Clone & Install

```bash
git clone <repo-url>
cd ICPR-2026

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
bash run.sh download
```

This downloads the training dataset to `data/train/` with the following structure:

```
data/train/
└── Scenario-A/
    └── {Brazilian,Mercosur}/
        └── track_XXXXX/
            ├── annotations.json   # {"plate_text": "AVL5215"}
            ├── lr-001.png … lr-005.png   # 5 low-res frames (~50×25 px)
            └── hr-001.png … hr-005.png   # 5 high-res frames (~60×30 px)
```

> **Total:** 18,000 train/val tracks + 2,000 test tracks.

If `run.sh download` is not available, download the dataset manually and place it at `data/train/`.

### 3. Train

```bash
WANDB_API_KEY=<your-key> python train.py
```

All parameters are configurable via environment variables:

| Variable | Default | Description |
|---|---|---|
| `WANDB_API_KEY` | — | Required for Weights & Biases |
| `BATCH_SIZE` | `4` | Micro-batch size |
| `GRAD_ACCUM` | `4` | Gradient accumulation steps |
| `EPOCHS` | `80` | Number of epochs |
| `LR` | `0.0003` | Learning rate |
| `NUM_WORKERS` | `4` | DataLoader workers |
| `USE_SR` | `1` | Enable LightSR module (0/1) |
| `DATA_ROOT` | `data/train` | Dataset path |

Example with custom settings:

```bash
WANDB_API_KEY=abc123 \
BATCH_SIZE=8 \
EPOCHS=100 \
NUM_WORKERS=4 \
USE_SR=1 \
  python train.py
```

### 4. Test / Evaluate

```bash
# Copy best model to test filename
cp best_model.pth best_model2.pth

# Run evaluation
python test.py
```

Results are saved to `test_results.json`:

```json
{
  "test_accuracy": 78.0,
  "char_accuracy": 94.2,
  "total_samples": 2000,
  "correct_samples": 1560
}
```

---

## Architecture

```
Input [B, T=5, 3, 64, 256]   ← 5 LR frames per track
    │
    │  Module 1: LightSR
    ▼
LR → 4× Upscale → [B*T, 3, 256, 1024]
    │
    │  Module 2: Recognition
    ▼
ConvNeXt Tiny (pretrained) → [B*T, 768, 8, 32]
    ↓
ConvProj (768→256) + Select center frame → [B, 256, 8, 32]
    ↓
AdaptiveAvgPool (W→1) + Permute → [B, 8, 256]
    ↓
BiLSTM (1 layer, 256 hidden) → [B, 8, 512]
    ↓
FC + LogSoftmax → [B, 8, 37]
```

**Total parameters:** ~31M (ConvNeXt backbone: 89%, LightSR: 1.6%, BiLSTM: 3.4%, FC head: 5.7%)

### Design Decisions

| Component | Decision | Reason |
|---|---|---|
| LightSR (not RealESRGAN) | 2 residual blocks + PixelShuffle | Lightweight SR, enough for text upscaling |
| Center frame selection | No fusion layer | All frames in a track are identical (plate is static) |
| BiLSTM (not Transformer) | 1 bidirectional layer | 7 characters is too short for Transformer; LSTM is lighter |
| No STN | Removed | License plate images are already relatively straight |

---

## Training Details

- **Gradual unfreezing:**
  - Phase 0 (ep 0–4): only FC head trains
  - Phase 1 (ep 5–9): LightSR + FC head train
  - Phase 2 (ep 10+): full training with 30% backbone freeze probability
- **Augmentation:** affine/elastic/perspective + brightness + blur/noise + coarse dropout
- **Scheduler:** OneCycleLR (30% warmup, cosine annealing)
- **Mixed precision:** AMP GradScaler on CUDA
- **CTC decoding:** greedy (beam_width=1) train/val; beam search (beam_width=5) + Brazilian format filter test

---

## Data Format

**Brazilian/Mercosur plates:** `AAA####` — 3 uppercase letters + 4 digits

Character set: `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ` (36 classes + 1 CTC blank)

---

## File Structure

```
ICPR-2026/
├── config.py              # All hyperparameters
├── train.py               # Training script
├── test.py                # Evaluation script
├── dataset.py             # Dataset loading (train/val/test)
├── transforms.py           # Data augmentation
├── utils.py               # CTC decode, format filter, seeding
├── models/
│   ├── crnn.py            # Full 2-module model
│   ├── light_sr.py        # Module 1: Lightweight SR
│   └── __init__.py
├── data/                  # Dataset (after download)
├── CLAUDE.md              # Developer notes
└── requirements.txt
```
