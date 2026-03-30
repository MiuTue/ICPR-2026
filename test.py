import os
import json
import torch
import torch.nn as nn
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
    print("\n" + "="*60)
    print("🧪 EVALUATING ON HELD-OUT TEST SET")
    print("="*60)
    print(f"   Test data : {Config.DATA_TEST}")

    # Test tracks (from TEST_TRACKS_FILE) live inside DATA_ROOT
    test_ds = TestDataset(root_dir=Config.DATA_ROOT)

    if len(test_ds) == 0:
        print("❌ Test loader rỗng!")
        return

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=TestDataset.collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.DEVICE.type == 'cuda'
    )

    # Model
    model = MultiFrameCRNN(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)

    if os.path.exists("best_model2.pth"):
        print(f"📂 Loading weights from 'best_model2.pth'...")
        checkpoint = torch.load("best_model2.pth", weights_only=True, map_location=Config.DEVICE)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        if missing_keys:
            print(f"   ⚠️ Missing keys : {missing_keys[:3]}")
        if unexpected_keys:
            print(f"   ⚠️ Unexpected keys: {unexpected_keys[:3]}")
        model.eval()
    else:
        print("❌ Không tìm thấy best_model2.pth — skipping inference.")
        return

    test_correct = 0
    test_total = 0
    test_char_correct = 0
    test_char_total = 0
    errors = []

    with torch.no_grad():
        for images, targets, target_lengths, labels_text in tqdm(test_loader, desc="Testing"):
            images = images.to(Config.DEVICE)
            preds = model(images)

            # Beam search + format filter for final evaluation
            decoded = decode_predictions(
                preds.log_softmax(2), Config.IDX2CHAR,
                beam_width=5, use_format_filter=True
            )

            for i in range(len(labels_text)):
                gt = labels_text[i]
                pred = decoded[i]

                if pred == gt:
                    test_correct += 1
                else:
                    if len(errors) < 20:
                        errors.append({'gt': gt, 'pred': pred})
                test_total += 1

                # Character-level accuracy
                for j in range(max(len(gt), len(pred))):
                    test_char_total += 1
                    if j < len(gt) and j < len(pred) and gt[j] == pred[j]:
                        test_char_correct += 1

    test_acc = (test_correct / test_total) * 100 if test_total > 0 else 0
    char_acc = (test_char_correct / test_char_total) * 100 if test_char_total > 0 else 0

    print(f"\n📊 TEST RESULTS:")
    print(f"   • Exact Match Accuracy : {test_acc:.2f}%  ({test_correct}/{test_total})")
    print(f"   • Character Accuracy   : {char_acc:.2f}%")

    if errors:
        print(f"\n🔍 Sample Errors (first 10):")
        for i, r in enumerate(errors[:10]):
            print(f"   {i+1}. GT: '{r['gt']}' | Pred: '{r['pred']}'")

    # Save results
    test_results = {
        'test_accuracy': test_acc,
        'char_accuracy': char_acc,
        'total_samples': test_total,
        'correct_samples': test_correct,
        'sample_errors': errors
    }
    with open('test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"\n💾 Results saved to 'test_results.json'")


if __name__ == "__main__":
    test_pipeline()
