import os
import torch


class Config:
    """Configuration class for LPR training."""

    # Data paths
    DATA_ROOT       = "data/train"         # all annotated tracks
    DATA_TEST       = "data/test"          # blind test (images only, no labels yet)
    TEST_TRACKS_FILE = "notebook/test_tracks.json"  # list of held-out test track names
    VAL_SPLIT_FILE  = "val_tracks.json"    # persisted train/val split

    # Image settings
    IMG_HEIGHT = 64
    IMG_WIDTH  = 256
    NUM_FRAMES = 8   # Intent: more temporal context; capped at 5 in dataset

    # Character set
    CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"

    # Training hyperparameters
    BATCH_SIZE    = 64   # reduce on CPU / low-VRAM GPUs
    LEARNING_RATE = 0.0003
    EPOCHS        = 80
    SEED          = 42
    NUM_WORKERS   = 4

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # STN / RealESRGAN
    USE_STN  = True    # spatial transformer for geometric correction
    USE_SR    = True   # RealESRGAN super-resolution
    SR_SCALE  = 2      # each RRDB block upsamples 2×

    # Character mappings (computed from CHARS)
    CHAR2IDX  = {char: idx + 1 for idx, char in enumerate(CHARS)}
    IDX2CHAR  = {idx + 1: char for idx, char in enumerate(CHARS)}
    NUM_CLASSES = len(CHARS) + 1  # +1 for CTC blank
