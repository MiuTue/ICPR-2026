import os
import torch


class Config:
    """Configuration class for LPR training.

    All values can be overridden via environment variables when running
    on remote machines (e.g. Vast.ai). Examples:

        export NUM_WORKERS=4
        export BATCH_SIZE=16
        export EPOCHS=80
        export WANDB_API_KEY=your_key_here
        python train.py
    """

    # ── Data paths ──────────────────────────────────────────────────────────
    DATA_ROOT        = os.getenv("DATA_ROOT",        "data/train")
    DATA_TEST        = os.getenv("DATA_TEST",        "data/test")
    TEST_TRACKS_FILE = os.getenv(
        "TEST_TRACKS_FILE", "notebook/test_tracks.json"
    )
    VAL_SPLIT_FILE   = os.getenv("VAL_SPLIT_FILE",   "val_tracks.json")

    # ── Image settings ──────────────────────────────────────────────────────
    IMG_HEIGHT = 64
    IMG_WIDTH  = 256
    NUM_FRAMES = 8   # Intent: capped at 5 in dataset._load_frames

    # ── Character set ───────────────────────────────────────────────────────
    CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"

    # ── Training hyperparameters ─────────────────────────────────────────────
    BATCH_SIZE    = int(os.getenv("BATCH_SIZE",    "64"))
    LEARNING_RATE = float(os.getenv("LR",          "0.0003"))
    EPOCHS        = int(os.getenv("EPOCHS",         "80"))
    SEED          = int(os.getenv("SEED",           "42"))
    NUM_WORKERS   = int(os.getenv("NUM_WORKERS",     "4"))   # 4 on Linux, 0 on macOS

    # ── Device ───────────────────────────────────────────────────────────────
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ── STN / RealESRGAN ────────────────────────────────────────────────────
    USE_STN  = os.getenv("USE_STN",  "1") == "1"
    USE_SR   = os.getenv("USE_SR",   "1") == "1"
    SR_SCALE = 2   # each RRDB block upscales 2× (total: 4×)

    # ── W&B ─────────────────────────────────────────────────────────────────
    WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")
    WANDB_PROJECT = os.getenv("WANDB_PROJECT", "icpr-2026-lpr")

    # ── Character mappings ──────────────────────────────────────────────────
    CHAR2IDX   = {char: idx + 1 for idx, char in enumerate(CHARS)}
    IDX2CHAR   = {idx + 1: char for idx, char in enumerate(CHARS)}
    NUM_CLASSES = len(CHARS) + 1   # +1 for CTC blank
