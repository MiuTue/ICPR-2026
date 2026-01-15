"""
Training script for Multi-Frame CRNN License Plate Recognition.

Usage:
    python train.py

The data directory should be configured in config.py (DATA_ROOT).
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# Support both running as module and direct script execution
try:
    from .config import Config
    from .dataset import AdvancedMultiFrameDataset
    from .models import MultiFrameCRNN
    from .utils import seed_everything, decode_predictions
except ImportError:
    from config import Config
    from dataset import AdvancedMultiFrameDataset
    from models import MultiFrameCRNN
    from utils import seed_everything, decode_predictions


def train_pipeline():
    """Main training pipeline."""
    seed_everything(Config.SEED)
    print(f"🚀 TRAINING START | Device: {Config.DEVICE}")
    
    # Check data directory
    if not os.path.exists(Config.DATA_ROOT):
        print(f"❌ LỖI: Sai đường dẫn DATA_ROOT: {Config.DATA_ROOT}")
        return

    # Create datasets
    train_ds = AdvancedMultiFrameDataset(Config.DATA_ROOT, mode='train', split_ratio=0.8)
    val_ds = AdvancedMultiFrameDataset(Config.DATA_ROOT, mode='val', split_ratio=0.8)
    
    if len(train_ds) == 0: 
        print("❌ Dataset Train rỗng!")
        return

    # Create data loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        collate_fn=AdvancedMultiFrameDataset.collate_fn, 
        num_workers=Config.NUM_WORKERS, 
        pin_memory=True
    )
    
    if len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=False, 
            collate_fn=AdvancedMultiFrameDataset.collate_fn, 
            num_workers=Config.NUM_WORKERS, 
            pin_memory=True
        )
    else:
        print("⚠️ CẢNH BÁO: Validation Set rỗng. Sẽ bỏ qua bước validate.")
        val_loader = None

    # Initialize model, loss, optimizer
    model = MultiFrameCRNN(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-3) # Tăng Weight Decay để điều hòa weights tốt hơn
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=Config.EPOCHS,
        pct_start=0.3, # Tăng thời gian warmup lên 30% tổng số bước
        div_factor=25.0, # Khởi đầu từ LR thấp hơn (max_lr / 25)
        final_div_factor=1000.0,
        anneal_strategy='cos'    
    )
    scaler = GradScaler()

    best_acc = 0.0
    
    # Training loop
    for epoch in range(Config.EPOCHS):
        # Stochastic Freezing logic
        if epoch < 5:
            model.freeze_backbone(True)
        else:
            # Sau epoch 5, áp dụng đóng băng ngẫu nhiên (ví dụ 30% xác suất)
            should_freeze = np.random.random() < 0.3
            model.freeze_backbone(should_freeze)
            
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{Config.EPOCHS}")
        for images, targets, target_lengths, _ in pbar:
            images = images.to(Config.DEVICE)
            targets = targets.to(Config.DEVICE)
            target_lengths = target_lengths.to(Config.DEVICE)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Apply Mixup with 30% probability
            use_mixup = np.random.random() < 0.15
            if use_mixup:
                # Prepare mixed data
                lam = np.random.beta(1.0, 1.0)
                index = torch.randperm(images.size(0)).to(Config.DEVICE)
                
                mixed_images = lam * images + (1 - lam) * images[index]
                
                # Prepare second set of targets for CTC
                y_list = torch.split(targets, target_lengths.tolist())
                y_list2 = [y_list[i] for i in index]
                targets2 = torch.cat(y_list2)
                target_lengths2 = target_lengths[index]
            else:
                mixed_images = images
                lam = 1.0

            with autocast('cuda'):
                preds = model(mixed_images)
                preds_permuted = preds.permute(1, 0, 2)
                input_lengths = torch.full(
                    size=(images.size(0),), 
                    fill_value=preds.size(1), 
                    dtype=torch.long
                )
                
                if use_mixup:
                    loss1 = criterion(preds_permuted, targets, input_lengths, target_lengths)
                    loss2 = criterion(preds_permuted, targets2, input_lengths, target_lengths2)
                    loss = lam * loss1 + (1 - lam) * loss2
                else:
                    loss = criterion(preds_permuted, targets, input_lengths, target_lengths)

            scaler_scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            
            # Giải phóng gradient bị quá lớn (Gradient Clipping)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Siết chặt clipping để tránh bùng nổ gradient
            
            scaler.step(optimizer)
            scaler.update()
            
            # Only step scheduler if optimizer actually stepped
            if scaler.get_scale() >= scaler_scale_before:
                scheduler.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]})
            
        avg_train_loss = epoch_loss / len(train_loader)

        # Validation
        val_acc = 0
        avg_val_loss = 0
        
        if val_loader:
            model.eval()
            val_loss = 0
            total_correct = 0
            total_samples = 0
            
            with torch.no_grad():
                for images, targets, target_lengths, labels_text in val_loader:
                    images = images.to(Config.DEVICE)
                    targets = targets.to(Config.DEVICE)
                    target_lengths = target_lengths.to(Config.DEVICE)
                    
                    preds = model(images)
                    
                    loss = criterion(
                        preds.permute(1, 0, 2), 
                        targets, 
                        torch.full((images.size(0),), preds.size(1), dtype=torch.long), 
                        target_lengths
                    )
                    val_loss += loss.item()
                    
                    # Use Greedy during training validation to save time.
                    # Only use Beam Search in final test.py
                    decoded = decode_predictions(preds, Config.IDX2CHAR, beam_width=1)
                    for i in range(len(labels_text)):
                        if decoded[i] == labels_text[i]:
                            total_correct += 1
                    total_samples += len(labels_text)

            avg_val_loss = val_loss / len(val_loader)
            val_acc = (total_correct / total_samples) * 100 if total_samples > 0 else 0
        
        print(f"Result: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f" -> ⭐ Saved Best Model! ({val_acc:.2f}%)")


if __name__ == "__main__":
    train_pipeline()
