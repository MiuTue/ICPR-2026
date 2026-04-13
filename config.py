import os
import torch


class Config:
    """Configuration class for LPR training."""
    
    # Data paths
    DATA_ROOT = "data/train"
    VAL_SPLIT_FILE = "val_tracks.json"
    
    # Image settings
    IMG_HEIGHT = 32
    IMG_WIDTH = 128
    HR_IMG_HEIGHT = 128   # Real-ESRGAN x4: 32*4=128
    HR_IMG_WIDTH = 512     # Real-ESRGAN x4: 128*4=512
    
    # Character set
    CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"
    
    # Training hyperparameters
    BATCH_SIZE = 64
    LR_RECOGNITION = 0.0003   # Transformer + FC + ConvProj (main)
    LR_SR = 5e-5             # Super-Resolution module
    LR_BACKBONE = 1e-5       # ConvNeXt Backbone
    
    MIXUP_PROB = 0.05        # Probability of using Mixup
    STOCHASTIC_FREEZE_PROB = 0.3 # Chance to unfreeze SR/Backbone each epoch after FREEZE_UNTIL_EPOCH

    EPOCHS = 50
    SEED = 42
    NUM_WORKERS = 4
    FREEZE_UNTIL_EPOCH = 5    # Initial freeze period
    LAMBDA_SR = 1.0           # Weight for SR loss (MSE)

    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Character mappings (computed from CHARS)
    CHAR2IDX = {char: idx + 1 for idx, char in enumerate(CHARS)}
    IDX2CHAR = {idx + 1: char for idx, char in enumerate(CHARS)}
    NUM_CLASSES = len(CHARS) + 1  # +1 for CTC blank
