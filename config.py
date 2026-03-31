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

    # ── Character set (Brazilian/Mercosur plates: 3 letters + 4 digits) ─────
    CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # ── Training hyperparameters ─────────────────────────────────────────────
    BATCH_SIZE     = int(os.getenv("BATCH_SIZE",    "4"))
    GRAD_ACCUM     = int(os.getenv("GRAD_ACCUM",     "4"))
    LEARNING_RATE  = float(os.getenv("LR",          "0.0003"))
    EPOCHS          = int(os.getenv("EPOCHS",         "80"))
    SEED            = int(os.getenv("SEED",           "42"))
    NUM_WORKERS     = int(os.getenv("NUM_WORKERS",     "4"))

    # ── Device ───────────────────────────────────────────────────────────────
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ── SR module ────────────────────────────────────────────────────────────
    USE_SR   = os.getenv("USE_SR",   "1") == "1"

    # ── W&B ─────────────────────────────────────────────────────────────────
    WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")
    WANDB_PROJECT = os.getenv("WANDB_PROJECT", "icpr-2026-lpr")

    # ── Character mappings ──────────────────────────────────────────────────
    CHAR2IDX   = {char: idx + 1 for idx, char in enumerate(CHARS)}
    IDX2CHAR   = {idx + 1: char for idx, char in enumerate(CHARS)}
    NUM_CLASSES = len(CHARS) + 1   # +1 for CTC blank
