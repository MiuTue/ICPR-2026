"""
Test script for Multi-Frame LPR.
Evaluates on held-out test set (5 LR frames/track → SR → recognition).

Usage:
    python test.py

The model (best_model2.pth) is loaded with strict=False loading.
"""

import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .config import Config
    from .dataset import TestDataset
    from .models import MultiFrameCRNN
    from .utils import seed_everything, decode_predictions
except ImportError:
    from config import Config
    from dataset import TestDataset
    from models import MultiFrameCRNN
    from utils import seed_everything, decode_predictions


def test_pipeline():
    seed_everything(Config.SEED)
    print("\n" + "=" * 60)
    print("TEST: 5 LR frames/track → SR → Recognition")
    print("=" * 60)

    # ── Test Dataset: chỉ chứa LR frames ──────────────────────────────────
    test_ds = TestDataset(root_dir=Config.DATA_ROOT)

    if len(test_ds) == 0:
        print("Test dataset empty!")
        return

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=TestDataset.collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.DEVICE.type == 'cuda'
    )
    print(f"   Test samples: {len(test_ds)}")

    # ── Model: Module 1 (SR) + Module 2 (Recognition) ─────────────────────
    model = MultiFrameCRNN(
        num_classes=Config.NUM_CLASSES,
        use_sr=Config.USE_SR,
    ).to(Config.DEVICE)

    if os.path.exists("best_model2.pth"):
        print(f"Loading weights from 'best_model2.pth'...")
        checkpoint = torch.load(
            "best_model2.pth",
            weights_only=True,
            map_location=Config.DEVICE
        )
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        if missing:
            print(f"   Missing keys: {missing[:3]}")
        if unexpected:
            print(f"   Unexpected keys: {unexpected[:3]}")
        model.eval()
    else:
        print("best_model2.pth not found — skipping inference.")
        return

    # ── Inference ─────────────────────────────────────────────────────────
    total_correct = 0
    total_samples = 0
    char_correct = 0
    char_total = 0
    errors = []

    with torch.no_grad():
        for images, _, _, labels_text in tqdm(test_loader, desc="Testing"):
            images = images.to(Config.DEVICE)

            # Module 1 (SR): LR → HR
            # Module 2: Recognition on SR frames
            preds = model(images, use_sr=True)  # Luôn dùng SR ✓

            # CTC Beam Search decode
            decoded = decode_predictions(
                preds, Config.IDX2CHAR,
                beam_width=5,
                use_format_filter=True,  # Brazilian plate filter
            )

            for gt, pred_text in zip(labels_text, decoded):
                if pred_text == gt:
                    total_correct += 1
                else:
                    if len(errors) < 20:
                        errors.append({'gt': gt, 'pred': pred_text})
                total_samples += 1

                # Character-level accuracy
                for j in range(max(len(gt), len(pred_text))):
                    char_total += 1
                    if j < len(gt) and j < len(pred_text) and gt[j] == pred_text[j]:
                        char_correct += 1

    # ── Results ───────────────────────────────────────────────────────────
    test_acc = (total_correct / total_samples * 100) if total_samples > 0 else 0
    char_acc = (char_correct / char_total * 100) if char_total > 0 else 0

    print(f"\n--- RESULTS ---")
    print(f"   Exact Match Accuracy : {test_acc:.2f}%  ({total_correct}/{total_samples})")
    print(f"   Character Accuracy   : {char_acc:.2f}%")

    if errors:
        print(f"\nSample Errors:")
        for i, r in enumerate(errors[:10]):
            print(f"   {i+1}. GT: '{r['gt']}' | Pred: '{r['pred']}'")

    # ── Save results ───────────────────────────────────────────────────────
    results = {
        'test_accuracy': test_acc,
        'char_accuracy': char_acc,
        'total_samples': total_samples,
        'correct_samples': total_correct,
        'sample_errors': errors,
    }
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to 'test_results.json'")


if __name__ == "__main__":
    test_pipeline()
