# LPR Baseline

Multi-Frame License Plate Recognition using Swin Transformer and Ref-Aware Temporal Fusion.

## Features

- **Swin Transformer Backbone**: Uses Swin-Tiny for efficient and powerful spatial feature extraction.
- **Multi-frame input**: Supports dynamic frame counts (default 5) for robust recognition.
- **Ref-Aware Fusion**: Learns attention weights relative to a reference frame for stable temporal fusion.
- **Transformer Encoder**: High-capacity sequence modeling using Transformer architecture.
- **Synthetic Degradation**: Augments images with blur, noise, and compression for robustness.
- **CTC Loss**: Handles variable-length license plate text effectively.

## Project Structure

```
lpr_baseline/
├── config.py           # Hyperparameters and paths
├── transforms.py       # Data augmentation pipeline
├── dataset.py          # Advanced multi-frame dataset loading
├── train.py            # Training script with AMP and Gradient Clipping
├── test.py             # Evaluation script
├── verify_fix.py       # Model sanity check script
└── models/
    ├── fusion.py       # RefAwareFusion, ChannelSpatialFusion, etc.
    └── crnn.py         # MultiFrameCRNN (Swin + Transformer)
```

## Installation

```bash
pip install -r requirements.txt
```

1. **Configure data path** in `config.py`:
   ```python
   DATA_ROOT = "path/to/your/data"
   ```

2. **Run training**:
   ```bash
   python train.py
   ```

3. **Verify model setup**:
   ```bash
   python verify_fix.py
   ```

## Data Format

Expected directory structure:
```
data/train/
├── Scenario-A/
│   └── track_00001/
│       ├── annotations.json    # {"plate_text": "ABC123"}
│       ├── lr-00.png          # Low-resolution frames
│       ├── lr-01.png
│       └── hr-00.png          # High-resolution frames (optional)
```

## Model Architecture

```
Input [Batch, T, 3, 32, 128]
    ↓
Swin Tiny Backbone → [B*T, 768, 1, 4]
    ↓
Conv Projection & Upsampling → [B*T, 512, 1, 16]
    ↓
Ref-Aware Fusion (Temporal) → [B, 512, 1, 16]
    ↓
Sequence Formatting → [B, 16, 512]
    ↓
Transformer Encoder (8 layers) → [B, 16, 512]
    ↓
FC + LogSoftmax → [B, 16, num_classes]
```

## License

MIT
