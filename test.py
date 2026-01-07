import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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

def test_pipeline():
    seed_everything(Config.SEED)
    print("\n" + "="*60)
    print("🧪 EVALUATING ON TEST SET...")
    print("="*60)

    # Load Dataset
    test_ds = AdvancedMultiFrameDataset(Config.DATA_ROOT, mode='test')
    
    if len(test_ds) == 0:
        print("❌ Test loader không có dữ liệu!")
        return

    test_loader = DataLoader(
        test_ds, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False,
        collate_fn=AdvancedMultiFrameDataset.collate_fn, 
        num_workers=Config.NUM_WORKERS, 
        pin_memory=True
    )

    # Initialize Model
    model = MultiFrameCRNN(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    
    if os.path.exists("best_model.pth"):
        # Load best model
        print(f"📂 Loading weights from 'best_model.pth'...")
        model.load_state_dict(torch.load("best_model.pth", weights_only=True, map_location=Config.DEVICE))
        model.eval()

        test_correct = 0
        test_total = 0
        test_char_correct = 0
        test_char_total = 0

        results = []  # Lưu kết quả để phân tích

        with torch.no_grad():
            for images, targets, target_lengths, labels_text in tqdm(test_loader, desc="Testing"):
                images = images.to(Config.DEVICE)
                preds = model(images)
                decoded = decode_predictions(torch.argmax(preds, dim=2), Config.IDX2CHAR)

                for i in range(len(labels_text)):
                    gt = labels_text[i]
                    pred = decoded[i]

                    # Exact match accuracy
                    if pred == gt:
                        test_correct += 1
                    test_total += 1

                    # Character-level accuracy
                    for j in range(max(len(gt), len(pred))):
                        test_char_total += 1
                        if j < len(gt) and j < len(pred) and gt[j] == pred[j]:
                            test_char_correct += 1

                    # Lưu một số kết quả sai để debug
                    if pred != gt and len(results) < 20:
                        results.append({'gt': gt, 'pred': pred})

        test_acc = (test_correct / test_total) * 100 if test_total > 0 else 0
        char_acc = (test_char_correct / test_char_total) * 100 if test_char_total > 0 else 0

        print(f"\n📊 TEST RESULTS:")
        print(f"   • Exact Match Accuracy: {test_acc:.2f}% ({test_correct}/{test_total})")
        print(f"   • Character Accuracy:   {char_acc:.2f}%")

        if results:
            print(f"\n🔍 Sample Errors (first 10):")
            for i, r in enumerate(results[:10]):
                print(f"   {i+1}. GT: '{r['gt']}' | Pred: '{r['pred']}'")

        # Lưu kết quả vào file
        test_results = {
            'test_accuracy': test_acc,
            'char_accuracy': char_acc,
            'total_samples': test_total,
            'correct_samples': test_correct,
            'sample_errors': results
        }
        with open('test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"\n💾 Results saved to 'test_results.json'")

    else:
        print("❌ Không tìm thấy best_model.pth!")

if __name__ == "__main__":
    test_pipeline()
