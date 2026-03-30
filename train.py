"""
Training script for Multi-Frame CRNN License Plate Recognition.

Usage:
    uv run python train.py

The data directory and model options are configured in config.py.
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


# Gradual unfreezing schedule
#   phase 0 (epochs 0-4)  : freeze backbone + freeze SR/STN  → train head only
#   phase 1 (epochs 5-9)  : freeze backbone + unfreeze SR/STN → train SR + head
#   phase 2 (epochs 10+)  : stochastic backbone freeze (30%)  → train all
FREEZE_SR_EPOCHS   = 5   # SR/STN trainable from epoch FREEZE_SR_EPOCHS
UNFREEZE_BACKBONE  = 10  # backbone fully trainable from epoch UNFREEZE_BACKBONE


def apply_freeze_schedule(model, epoch):
    """Apply gradual unfreezing schedule per epoch."""
    if epoch < FREEZE_SR_EPOCHS:
        # Phase 0: everything frozen except transformer + fusion + fc
        model.freeze_backbone(True)
        model.freeze_sr(True)
        phase = 0
    elif epoch < UNFREEZE_BACKBONE:
        # Phase 1: backbone still frozen, SR/STN now trainable
        model.freeze_backbone(True)
        model.freeze_sr(False)
        phase = 1
    else:
        # Phase 2: backbone with stochastic freeze
        model.freeze_sr(False)
        should_freeze = np.random.random() < 0.3
        model.freeze_backbone(should_freeze)
        phase = 2
    return phase


def train_pipeline():
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"🚀 TRAINING START | Device: {device}")
    print(f"   USE_STN : {Config.USE_STN}  |  USE_SR : {Config.USE_SR}")
    print(f"   Freeze schedule: SR unlocked @ ep{FREEZE_SR_EPOCHS}, "
          f"backbone unlocked @ ep{UNFREEZE_BACKBONE}")

    if not os.path.exists(Config.DATA_ROOT):
        print(f"❌ LỖI: Sai đường dẫn DATA_ROOT: {Config.DATA_ROOT}")
        return

    # Datasets
    train_ds = AdvancedMultiFrameDataset(Config.DATA_ROOT, mode='train', split_ratio=0.8)
    val_ds   = AdvancedMultiFrameDataset(Config.DATA_ROOT, mode='val',   split_ratio=0.8)

    if len(train_ds) == 0:
        print("❌ Dataset Train rỗng!")
        return

    pin_mem = device.type == 'cuda'
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=AdvancedMultiFrameDataset.collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=pin_mem
    )
    val_loader = (
        DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False,
                   collate_fn=AdvancedMultiFrameDataset.collate_fn,
                   num_workers=Config.NUM_WORKERS, pin_memory=pin_mem)
        if len(val_ds) > 0 else None
    )

    # Model
    model = MultiFrameCRNN(
        num_classes=Config.NUM_CLASSES,
        use_stn=Config.USE_STN,
        use_sr=Config.USE_SR,
    ).to(device)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=Config.EPOCHS,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0,
        anneal_strategy='cos'
    )
    scaler = GradScaler() if device.type == 'cuda' else None

    best_acc = 0.0

    for epoch in range(Config.EPOCHS):
        phase = apply_freeze_schedule(model, epoch)

        model.train()
        epoch_loss = 0

        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{Config.EPOCHS} [phase{phase}]")
        for images, targets, target_lengths, _ in pbar:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad(set_to_none=True)

            # Mixup (15% chance)
            use_mixup = np.random.random() < 0.15
            if use_mixup:
                lam = np.random.beta(1.0, 1.0)
                idx = torch.randperm(images.size(0)).to(device)
                mixed = lam * images + (1 - lam) * images[idx]

                y_list = torch.split(targets, target_lengths.tolist())
                targets2 = torch.cat([y_list[i] for i in idx])
                target_lengths2 = target_lengths[idx]
            else:
                mixed = images
                lam = 1.0

            # Forward pass — choose autocast context
            if device.type == 'cuda' and scaler is not None:
                with autocast('cuda'):
                    preds = model(mixed)
                    preds_permuted = preds.permute(1, 0, 2)
                    input_lengths = torch.full(
                        (images.size(0),), preds.size(1), dtype=torch.long, device=device
                    )
                    if use_mixup:
                        loss1 = criterion(preds_permuted, targets,  input_lengths, target_lengths)
                        loss2 = criterion(preds_permuted, targets2, input_lengths, target_lengths2)
                        loss = lam * loss1 + (1 - lam) * loss2
                    else:
                        loss = criterion(preds_permuted, targets, input_lengths, target_lengths)

                scaler_scale_before = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scaler_scale_before:
                    scheduler.step()
            else:
                # CPU path — no autocast/scaler; do NOT use torch.no_grad()
                preds = model(mixed)
                preds_permuted = preds.permute(1, 0, 2)
                input_lengths = torch.full(
                    (images.size(0),), preds.size(1), dtype=torch.long, device=device
                )
                if use_mixup:
                    loss1 = criterion(preds_permuted, targets,  input_lengths, target_lengths)
                    loss2 = criterion(preds_permuted, targets2, input_lengths, target_lengths2)
                    loss = lam * loss1 + (1 - lam) * loss2
                else:
                    loss = criterion(preds_permuted, targets, input_lengths, target_lengths)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})

        avg_train_loss = epoch_loss / len(train_loader)

        # Validation
        val_acc = 0.0
        avg_val_loss = 0.0
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            total_correct = 0
            total_samples = 0

            with torch.no_grad():
                for images, targets, target_lengths, labels_text in val_loader:
                    images = images.to(device)
                    targets = targets.to(device)
                    target_lengths = target_lengths.to(device)

                    preds = model(images)
                    input_lengths = torch.full(
                        (images.size(0),), preds.size(1), dtype=torch.long, device=device
                    )
                    loss = criterion(
                        preds.permute(1, 0, 2),
                        targets, input_lengths, target_lengths
                    )
                    val_loss += loss.item()

                    decoded = decode_predictions(preds, Config.IDX2CHAR, beam_width=1)
                    for gt, pred in zip(labels_text, decoded):
                        if pred == gt:
                            total_correct += 1
                        total_samples += 1

            avg_val_loss = val_loss / len(val_loader)
            val_acc = (total_correct / total_samples) * 100 if total_samples > 0 else 0.0

        print(f"Result: Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"   -> ⭐ Saved Best Model! ({val_acc:.2f}%)")


if __name__ == "__main__":
    train_pipeline()
