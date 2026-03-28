# ICPR 2026 - End-to-End Super-Resolution & License Plate Recognition (LPR)

## Project Overview
This project implements an End-to-End pipeline for License Plate Recognition (LPR) from Low-Resolution images, as specified in the `Kế hoạch nghiên cứu báo cáo ICPC.pdf`. It combines a dedicated Super-Resolution module with a CNN-Transformer recognition head.

### Core Technologies
- **Environment Management:** `uv` (use `uv run <script.py>` for execution)
- **Framework:** PyTorch 2.11.0
- **Super-Resolution:** EDSR-style `SRModule` with Residual Blocks and PixelShuffle upsampling (2x).
- **Recognition Backbone:** Swin Transformer Tiny.
- **Sequence Modeling:** Transformer Encoder (8 layers).
- **Loss Functions:** 
  - `MSELoss`: For Super-Resolution quality (SR vs HR).
  - `CTCLoss`: For character recognition.
- **Data Augmentation:** Albumentations (including synthetic degradation for LR simulation).

### Architecture
1.  **SR Module:** Upscales LR (32x128) -> SR (64x256).
2.  **Swin Backbone:** Extracts features from the SR image.
3.  **Conv Projection:** Maps features to a 512-d sequence.
4.  **Transformer Encoder:** Processes the sequence for character dependencies.
5.  **FC Head:** Output log-probabilities for CTC.

---

## Building and Running

### Setup
Ensure `uv` is installed, then sync dependencies:
```bash
uv sync
```

### Configuration
Update `DATA_ROOT` in `config.py` to point to your ICPC dataset:
```python
DATA_ROOT = "path/to/your/data"
```

### Key Commands
- **Training:**
  ```bash
  uv run train.py
  ```
  Trains the model with multi-task loss (SR + CTC).
- **Testing:**
  ```bash
  uv run test.py
  ```
  Evaluates the model on the test set.

---

## Project Structure
- `config.py`: Configuration for LR/HR sizes, character set, and paths.
- `dataset.py`: `EndToEndDataset` handles (LR, HR, Label) triplets.
- `models/`:
  - `crnn.py`: Contains `SRModule`, `PositionalEncoding`, and `EndToEndLPR`.
- `train.py`: Multi-task training loop.
- `test.py`: Evaluation logic.
- `transforms.py`: Augmentation and normalization pipelines.

---

## Development Conventions
- **Multi-task Loss:** Total Loss = `CTC_Loss + lambda_sr * SR_Loss`.
- **Reproducibility:** Seed is fixed via `utils.seed_everything`.
- **Inference:** The model expects a single LR image and returns both the SR image and text predictions.
