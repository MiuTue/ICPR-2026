"""
Training script for End-to-End License Plate Recognition with Super-Resolution.
Combined CTC Loss (Recognition) and MSE Loss (Super-Resolution).
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
    from .dataset import EndToEndDataset
    from .models.crnn_realesrgan import EndToEndLPR
    from .utils import seed_everything, decode_predictions
except ImportError:
    from config import Config
    from dataset import EndToEndDataset
    from models.crnn_realesrgan import EndToEndLPR
    from utils import seed_everything, decode_predictions


def train_pipeline():
    """Main training pipeline for End-to-End LPR."""
    seed_everything(Config.SEED)
    torch.backends.cudnn.benchmark = True # Tăng tốc cho các phép toán Convolution
    print(f"🚀 END-TO-END TRAINING START | Device: {Config.DEVICE}")
    print(f"📊 LRs: Rec={Config.LR_RECOGNITION}, SR={Config.LR_SR}, BB={Config.LR_BACKBONE} | SR_Weight={Config.LAMBDA_SR}")
    
    # Check data directory
    if not os.path.exists(Config.DATA_ROOT):
        print(f"❌ LỖI: Sai đường dẫn DATA_ROOT: {Config.DATA_ROOT}")
        return

    # Create datasets
    train_ds = EndToEndDataset(Config.DATA_ROOT, mode='train', split_ratio=0.8)
    val_ds = EndToEndDataset(Config.DATA_ROOT, mode='val', split_ratio=0.8)
    
    if len(train_ds) == 0: 
        print("❌ Dataset Train rỗng!")
        return

    # Create data loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        collate_fn=EndToEndDataset.collate_fn, 
        num_workers=Config.NUM_WORKERS, 
        pin_memory=True
    )
    
    if len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=False, 
            collate_fn=EndToEndDataset.collate_fn, 
            num_workers=Config.NUM_WORKERS, 
            pin_memory=True
        )
    else:
        print("⚠️ CẢNH BÁO: Validation Set rỗng. Sẽ bỏ qua bước validate.")
        val_loader = None

    # Initialize model
    model = EndToEndLPR(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    model.load_state_dict(torch.load("best_model.pth"))
    
    # Dual Losses
    criterion_ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    criterion_sr = nn.MSELoss()
    
    # Multi-group Optimizer: Specific LRs per component
    optimizer = optim.AdamW([
        {'params': model.sr_module.parameters(),    'lr': Config.LR_SR},
        {'params': model.backbone.parameters(),     'lr': Config.LR_BACKBONE},
        {'params': model.conv_proj.parameters(),    'lr': Config.LR_RECOGNITION},
        {'params': model.transformer.parameters(), 'lr': Config.LR_RECOGNITION},
        {'params': model.fc.parameters(),          'lr': Config.LR_RECOGNITION},
    ], weight_decay=1e-3)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[Config.LR_SR, Config.LR_BACKBONE, Config.LR_RECOGNITION, Config.LR_RECOGNITION, Config.LR_RECOGNITION],
        steps_per_epoch=len(train_loader),
        epochs=Config.EPOCHS,
        pct_start=0.16, # Đẩy max learning rate đến sớm ở epoch 5 (0.1 * 50) thay vì đợi đến epoch 15
        div_factor=25.0, 
        final_div_factor=1000.0,
        anneal_strategy='cos'    
    )
    scaler = GradScaler()

    best_acc = 0.0
    
    # Training loop
    for epoch in range(Config.EPOCHS):
        # Freeze logic
        if epoch < Config.FREEZE_UNTIL_EPOCH:
            # Ở những epoch đầu, Freeze Backbone (Pre-trained) để Head có thời gian làm quen với việc SR ảnh
            model.set_freeze_mode(False)
        else:
            # Mở khoá vĩnh viễn Backbone để finetune toàn hệ thống
            model.set_freeze_mode(True)
            if epoch == Config.FREEZE_UNTIL_EPOCH:
                print(f" 🔓 [Epoch {epoch+1}] Full Pipeline Unfrozen for fine-tuning")
            
        model.train()
        epoch_ctc_loss = 0
        epoch_sr_loss = 0
        epoch_total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{Config.EPOCHS}")
        for lr_images, hr_images, targets, target_lengths, labels_text, _ in pbar:
            lr_images = lr_images.to(Config.DEVICE)
            hr_images = hr_images.to(Config.DEVICE)
            targets = targets.to(Config.DEVICE)
            target_lengths = target_lengths.to(Config.DEVICE)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Multi-task Mixup logic
            use_mixup = np.random.random() < Config.MIXUP_PROB
            if use_mixup:
                lam = np.random.beta(1.0, 1.0)
                index = torch.randperm(lr_images.size(0)).to(Config.DEVICE)
                
                mixed_lr = lam * lr_images + (1 - lam) * lr_images[index]
                mixed_hr = lam * hr_images + (1 - lam) * hr_images[index]
                
                # Intertwine targets for CTC loss Calculation
                y_list = torch.split(targets, target_lengths.tolist())
                y_list2 = [y_list[i] for i in index]
                targets2 = torch.cat(y_list2)
                target_lengths2 = target_lengths[index]
            else:
                mixed_lr = lr_images
                mixed_hr = hr_images
                lam = 1.0

            with autocast('cuda'):
                # Forward: [B, 3, 128, 512], [B, 16, num_classes]
                sr_img, logits = model(mixed_lr)
                
                # 1. CTC Loss (Apply log_softmax here as EndToEndLPR returns raw logits)
                log_probs = logits.log_softmax(2).permute(1, 0, 2) # [T, B, C]
                input_lengths = torch.full(
                    size=(lr_images.size(0),), 
                    fill_value=logits.size(1), 
                    dtype=torch.long
                )
                
                if use_mixup:
                    loss_ctc = lam * criterion_ctc(log_probs, targets, input_lengths, target_lengths) + \
                               (1 - lam) * criterion_ctc(log_probs, targets2, input_lengths, target_lengths2)
                else:
                    loss_ctc = criterion_ctc(log_probs, targets, input_lengths, target_lengths)
                
                # 2. SR Loss (MSE)
                loss_sr = criterion_sr(sr_img, mixed_hr)
                
                # Total multi-task loss
                loss = loss_ctc + Config.LAMBDA_SR * loss_sr

            scaler_scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            
            # Gradient Clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            if scaler.get_scale() >= scaler_scale_before:
                scheduler.step()
            
            epoch_ctc_loss += loss_ctc.item()
            epoch_sr_loss += loss_sr.item()
            epoch_total_loss += loss.item()
            
            # Tách biệt hiển thị ctc và sr loss trên thanh tiến trình
            pbar.set_postfix({
                'ctc': f"{loss_ctc.item():.4f}", 
                'sr': f"{loss_sr.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[2]:.2e}" # show recognition LR
            })
            
        avg_train_loss = epoch_total_loss / len(train_loader)
        avg_ctc_loss = epoch_ctc_loss / len(train_loader)
        avg_sr_loss = epoch_sr_loss / len(train_loader)

        # Validation
        val_acc = 0
        avg_val_loss = 0
        
        if val_loader:
            model.eval()
            total_val_loss = 0
            total_val_ctc = 0
            total_val_sr = 0
            total_correct = 0
            total_samples = 0
            
            with torch.no_grad():
                for lr_images, hr_images, targets, target_lengths, labels_text, _ in val_loader:
                    lr_images = lr_images.to(Config.DEVICE)
                    hr_images = hr_images.to(Config.DEVICE)
                    targets = targets.to(Config.DEVICE)
                    target_lengths = target_lengths.to(Config.DEVICE)
                    
                    sr_img, logits = model(lr_images)
                    
                    loss_ctc = criterion_ctc(
                        logits.log_softmax(2).permute(1, 0, 2), 
                        targets, 
                        torch.full((lr_images.size(0),), logits.size(1), dtype=torch.long), 
                        target_lengths
                    )
                    loss_sr = criterion_sr(sr_img, hr_images)
                    
                    combined_val_loss = loss_ctc + Config.LAMBDA_SR * loss_sr
                    total_val_loss += combined_val_loss.item()
                    total_val_ctc += loss_ctc.item()
                    total_val_sr += loss_sr.item()
                    
                    # Recognition Accuracy
                    decoded = decode_predictions(logits.argmax(2), Config.IDX2CHAR)
                    for i in range(len(labels_text)):
                        if decoded[i].strip() == labels_text[i].strip():
                            total_correct += 1
                    total_samples += len(labels_text)

            avg_val_loss = total_val_loss / len(val_loader)
            avg_val_ctc = total_val_ctc / len(val_loader)
            avg_val_sr = total_val_sr / len(val_loader)
            val_acc = (total_correct / total_samples) * 100 if total_samples > 0 else 0
        
        print(f"Result Ep {epoch+1}:")
        print(f"  - Train: Loss={avg_train_loss:.4f} (CTC={avg_ctc_loss:.4f}, SR={avg_sr_loss:.4f})")
        print(f"  - Val  : Loss={avg_val_loss:.4f} (CTC={avg_val_ctc:.4f}, SR={avg_val_sr:.4f}) | Acc={val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f" -> ⭐ Saved Best Model! ({val_acc:.2f}%)")


if __name__ == "__main__":
    train_pipeline()