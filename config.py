import os
import torch


class Config:
    """Configuration class for LPR training."""
    
    # Data paths
    DATA_ROOT = "data/train"
    VAL_SPLIT_FILE = "val_tracks.json"
    
    # Image settings
    IMG_HEIGHT = 64
    IMG_WIDTH = 256
    NUM_FRAMES = 8 # Tăng từ 5 lên 8 để có thêm ngữ cảnh thời gian
    
    # Character set
    CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"
    
    # Training hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0003 # Hạ LR xuống 0.0003 để triệt tiêu lỗi NaN loss hoàn toàn
    EPOCHS = 80 # Giữ nguyên 80 epochs cho lộ trình hội tụ ổn định
    SEED = 42
    NUM_WORKERS = 4
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Character mappings (computed from CHARS)
    CHAR2IDX = {char: idx + 1 for idx, char in enumerate(CHARS)}
    IDX2CHAR = {idx + 1: char for idx, char in enumerate(CHARS)}
    NUM_CLASSES = len(CHARS) + 1  # +1 for CTC blank
