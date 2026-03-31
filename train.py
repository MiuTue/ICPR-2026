"""
Training script for Multi-Frame LPR (2-Module Pipeline).

    Module 1: LightSR — upscale LR → HR (4×)
    Module 2: ConvNeXt + BiLSTM + CTC — recognition

Usage:
    WANDB_API_KEY=<your-key> python train.py

All params configurable via config.py and environment variables.
"""

import gc
import os
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

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


# ── Freeze schedule ──────────────────────────────────────────────────────────
FREEZE_SR_EPOCHS   = 5    # SR trainable from this epoch
UNFREEZE_BACKBONE   = 10   # backbone fully trainable from this epoch

# ── W&B logging ──────────────────────────────────────────────────────────────
LOG_STEP_EVERY    = 10
LOG_IMAGES_EVERY  = 50


def apply_freeze_schedule(model, epoch):
    """
    Gradual unfreezing schedule:
        Phase 0 (ep 0-4):  freeze SR + backbone + LSTM  → only FC head trains
        Phase 1 (ep 5-9):  unfreeze SR, keep backbone + LSTM frozen
        Phase 2 (ep 10+):  full training (SR + backbone stochastic, LSTM always on)
    """
    if epoch < FREEZE_SR_EPOCHS:
        # Phase 0: train only FC head
        model.freeze_sr(freeze=True)
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.lstm.parameters():
            param.requires_grad = False
        phase = 0
    elif epoch < UNFREEZE_BACKBONE:
        # Phase 1: unfreeze SR, backbone + LSTM still frozen
        model.freeze_sr(freeze=False)
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.lstm.parameters():
            param.requires_grad = False
        phase = 1
    else:
        # Phase 2: full training — SR + LSTM always on, backbone stochastic
        model.freeze_sr(freeze=False)
        for param in model.lstm.parameters():
            param.requires_grad = True
        should_freeze = np.random.random() < 0.3
        for param in model.backbone.parameters():
            param.requires_grad = not should_freeze
        phase = 2
    return phase


def _count_trainable(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return trainable, total


def _grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def train_pipeline():
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # ── W&B init ─────────────────────────────────────────────────────────────
    if Config.WANDB_API_KEY:
        wandb.login(key=Config.WANDB_API_KEY)

    run_name = (
        f"lpr-{datetime.datetime.now().strftime('%m%d-%H%M')}"
        f"-sr{int(Config.USE_SR)}"
        f"-bs{Config.BATCH_SIZE}x{Config.GRAD_ACCUM}"
    )
    wandb.init(
        project=Config.WANDB_PROJECT,
        name=run_name,
        config={
            # Architecture
            "architecture": "LightSR + ConvNeXt_Tiny + BiLSTM + CTC",
            "d_model": 256,
            "num_classes": Config.NUM_CLASSES,
            "use_sr": Config.USE_SR,
            # Data
            "img_height": Config.IMG_HEIGHT,
            "img_width": Config.IMG_WIDTH,
            "train_samples": None,
            "val_samples": None,
            # Training
            "batch_size": Config.BATCH_SIZE,
            "grad_accum": Config.GRAD_ACCUM,
            "effective_batch_size": Config.BATCH_SIZE * Config.GRAD_ACCUM,
            "learning_rate": Config.LEARNING_RATE,
            "epochs": Config.EPOCHS,
            "seed": Config.SEED,
            "freeze_sr_epochs": FREEZE_SR_EPOCHS,
            "unfreeze_backbone_epoch": UNFREEZE_BACKBONE,
            "mixup_prob": 0.15,
            "backbone_freeze_prob": 0.3,
            "weight_decay": 1e-3,
            "grad_clip": 1.0,
            "scheduler": "OneCycleLR(pct_start=0.3,cos)",
        },
        notes="LightSR + ConvNeXt Tiny + BiLSTM CRNN",
        tags=["lpr", "ctc", "lightsr", "convnext"],
    )
    print(f"W&B run: {wandb.run.url}")

    # ── Data ─────────────────────────────────────────────────────────────────
    if not os.path.exists(Config.DATA_ROOT):
        print(f"DATA_ROOT not found: {Config.DATA_ROOT}")
        wandb.finish(exit_code=1)
        return

    train_ds = AdvancedMultiFrameDataset(
        Config.DATA_ROOT, mode='train', split_ratio=0.8
    )
    val_ds   = AdvancedMultiFrameDataset(
        Config.DATA_ROOT, mode='val',   split_ratio=0.8
    )

    if len(train_ds) == 0:
        print("Train dataset empty!")
        wandb.finish(exit_code=1)
        return

    wandb.config.update({
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
    }, allow_val_change=True)

    pin_mem = device.type == 'cuda'
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=AdvancedMultiFrameDataset.collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=pin_mem,
        drop_last=True,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            collate_fn=AdvancedMultiFrameDataset.collate_fn,
            num_workers=Config.NUM_WORKERS,
            pin_memory=pin_mem,
        )
        if len(val_ds) > 0 else None
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = MultiFrameCRNN(
        num_classes=Config.NUM_CLASSES,
        use_sr=Config.USE_SR,
    ).to(device)

    trainable_p, total_p = _count_trainable(model)
    print(f"Trainable: {trainable_p:,} / {total_p:,}  ({100*trainable_p/total_p:.1f}%)")
    wandb.config.update({
        "total_params": total_p,
        "trainable_params": trainable_p,
    }, allow_val_change=True)

    wandb.watch(model, log="gradients", log_freq=LOG_STEP_EVERY * 10)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=1e-3,
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=Config.EPOCHS,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0,
        anneal_strategy='cos',
    )
    scaler = GradScaler() if device.type == 'cuda' else None

    best_acc         = 0.0
    global_step      = 0
    accum_step       = 0
    epoch_start_time = time.time()

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(Config.EPOCHS):
        phase = apply_freeze_schedule(model, epoch)
        phase_names = {0: "head-only", 1: "sr+head", 2: "full"}

        model.train()
        epoch_loss  = 0.0
        epoch_steps = 0
        accum_step  = 0
        mixup_count = 0

        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{Config.EPOCHS} [{phase_names[phase]}]")
        for batch_idx, (images, targets, target_lengths, _) in enumerate(pbar):
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            if accum_step == 0:
                optimizer.zero_grad(set_to_none=True)

            # ── Mixup ──────────────────────────────────────────────────────
            use_mixup = np.random.random() < 0.15
            if use_mixup:
                lam = np.random.beta(1.0, 1.0)
                idx = torch.randperm(images.size(0)).to(device)
                mixed = lam * images + (1 - lam) * images[idx]
                y_list = torch.split(targets, target_lengths.tolist())
                targets2 = torch.cat([y_list[i] for i in idx])
                target_lengths2 = target_lengths[idx]
                mixup_count += 1
            else:
                mixed = images
                lam = 1.0

            # ── Forward ─────────────────────────────────────────────────────
            if device.type == 'cuda' and scaler is not None:
                with autocast('cuda'):
                    preds = model(mixed, use_sr=Config.USE_SR)
                    preds_permuted = preds.permute(1, 0, 2)
                    input_lens = torch.full(
                        (images.size(0),), preds.size(1),
                        dtype=torch.long, device=device
                    )
                    if use_mixup:
                        loss1 = criterion(preds_permuted, targets,   input_lens, target_lengths)
                        loss2 = criterion(preds_permuted, targets2,  input_lens, target_lengths2)
                        loss = lam * loss1 + (1 - lam) * loss2
                    else:
                        loss = criterion(preds_permuted, targets, input_lens, target_lengths)
                    loss_scaled = loss / Config.GRAD_ACCUM
                    scaler.scale(loss_scaled).backward()
            else:
                preds = model(mixed, use_sr=Config.USE_SR)
                preds_permuted = preds.permute(1, 0, 2)
                input_lens = torch.full(
                    (images.size(0),), preds.size(1),
                    dtype=torch.long, device=device
                )
                if use_mixup:
                    loss1 = criterion(preds_permuted, targets,  input_lens, target_lengths)
                    loss2 = criterion(preds_permuted, targets2, input_lens, target_lengths2)
                    loss = lam * loss1 + (1 - lam) * loss2
                else:
                    loss = criterion(preds_permuted, targets, input_lens, target_lengths)
                (loss / Config.GRAD_ACCUM).backward()

            accum_step  += 1
            epoch_steps += 1
            global_step += 1

            # ── Optimizer step: after GRAD_ACCUM micro-batches ──────────────
            if accum_step % Config.GRAD_ACCUM == 0:
                if device.type == 'cuda' and scaler is not None:
                    scaler.unscale_(optimizer)
                    g_norm = _grad_norm(model)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() >= 1.0:
                        scheduler.step()
                    torch.cuda.empty_cache()
                else:
                    g_norm = _grad_norm(model)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()

                accum_step = 0

                # Per-step W&B logging
                if LOG_STEP_EVERY > 0:
                    wandb.log({
                        "step/loss":      loss.item(),
                        "step/lr":        scheduler.get_last_lr()[0],
                        "step/grad_norm": g_norm,
                        "step/phase":     phase,
                        "step/epoch":      epoch + 1,
                        "step/global_step": global_step,
                    }, step=global_step)

                # Per-step prediction table
                if LOG_IMAGES_EVERY > 0 and epoch_steps % LOG_IMAGES_EVERY == 0:
                    model.eval()
                    with torch.no_grad():
                        preds_raw = model(images[:8], use_sr=Config.USE_SR)
                        decoded = decode_predictions(
                            preds_raw, Config.IDX2CHAR, beam_width=1
                        )
                    model.train()
                    _log_predictions_table(images, list(_)[:8], decoded, global_step)

                epoch_loss += loss.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr':   f'{scheduler.get_last_lr()[0]:.2e}',
            })

        # Drain remaining accumulated gradient
        if accum_step > 0:
            if device.type == 'cuda' and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                torch.cuda.empty_cache()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        avg_train_loss = epoch_loss / max(epoch_steps // max(Config.GRAD_ACCUM, 1), 1)

        if device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

        # ── Validation ───────────────────────────────────────────────────────
        avg_val_loss     = 0.0
        val_acc          = 0.0
        val_char_acc     = 0.0
        val_sample_errors = []

        if val_loader is not None:
            val_loss     = 0.0
            val_steps    = 0
            total_correct = 0
            total_samples = 0
            val_char_correct = 0
            val_char_total   = 0
            model.eval()

            with torch.no_grad():
                for images, targets, target_lengths, labels_text in tqdm(
                    val_loader, desc=f"Ep {epoch+1} [val]"
                ):
                    images = images.to(device)
                    targets = targets.to(device)
                    target_lengths = target_lengths.to(device)

                    # Validation: LR → SR → Recognition (same as test)
                    preds = model(images, use_sr=Config.USE_SR)
                    input_lens = torch.full(
                        (images.size(0),), preds.size(1),
                        dtype=torch.long, device=device
                    )
                    loss = criterion(
                        preds.permute(1, 0, 2), targets, input_lens, target_lengths
                    )
                    val_loss += loss.item()
                    val_steps += 1

                    decoded = decode_predictions(
                        preds, Config.IDX2CHAR, beam_width=1
                    )
                    for gt, pred_text in zip(labels_text, decoded):
                        if pred_text == gt:
                            total_correct += 1
                        else:
                            if len(val_sample_errors) < 5:
                                val_sample_errors.append({"gt": gt, "pred": pred_text})
                        total_samples += 1

                        for j in range(max(len(gt), len(pred_text))):
                            val_char_total += 1
                            if j < len(gt) and j < len(pred_text) and gt[j] == pred_text[j]:
                                val_char_correct += 1

                    del preds, input_lens, loss, decoded
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                        gc.collect()

            avg_val_loss     = val_loss / max(val_steps, 1)
            val_acc           = (total_correct / total_samples * 100) if total_samples > 0 else 0.0
            val_char_acc      = (val_char_correct / val_char_total * 100) if val_char_total > 0 else 0.0
            model.train()

        epoch_duration = time.time() - epoch_start_time

        wandb.log({
            "epoch":                  epoch + 1,
            "epoch/train_loss":       avg_train_loss,
            "epoch/val_loss":         avg_val_loss,
            "epoch/val_acc":          val_acc,
            "epoch/val_char_acc":     val_char_acc,
            "epoch/learning_rate":    scheduler.get_last_lr()[0],
            "epoch/phase":            phase,
            "epoch/phase_name":       phase_names[phase],
            "epoch/best_val_acc":     best_acc,
            "epoch/duration_sec":     epoch_duration,
            "epoch/mixup_batches":    mixup_count,
            "epoch/steps":            epoch_steps,
        }, step=global_step)

        print(
            f"Result: Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%  "
            f"| Char Acc: {val_char_acc:.2f}%  | {epoch_duration:.0f}s"
        )

        # ── Save best model + W&B artifact ─────────────────────────────────
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"   -> Saved Best Model! ({val_acc:.2f}%)")
            artifact = wandb.Artifact(
                f"best-model-{wandb.run.id}",
                type="model",
                metadata={"val_acc": val_acc, "epoch": epoch + 1},
            )
            artifact.add_file("best_model.pth")
            wandb.log_artifact(artifact, aliases=["best"])

        epoch_start_time = time.time()

    print(f"\nTraining complete. Best Val Acc: {best_acc:.2f}%")
    print(f"W&B URL: {wandb.run.url}")
    wandb.finish()


def _log_predictions_table(images, labels_text, preds_list, step, n_samples=8):
    """Log a W&B table with input frames + ground-truth vs predicted text."""
    n = min(n_samples, len(labels_text))
    rows = []
    for i in range(n):
        frame = images[i, images.size(1) // 2].cpu()
        rows.append([
            wandb.Image(frame, caption=f"GT: {labels_text[i]}"),
            labels_text[i],
            preds_list[i],
            "OK" if preds_list[i] == labels_text[i] else "X",
        ])
    columns = ["Frame", "Ground Truth", "Prediction", "Match"]
    table = wandb.Table(data=rows, columns=columns)
    wandb.log({f"predictions/step_{step}": table}, step=step)


if __name__ == "__main__":
    train_pipeline()
