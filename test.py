import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .config import Config
    from .dataset import EndToEndDataset
    from .models.crnn import EndToEndLPR
    from .utils import seed_everything, decode_predictions
except ImportError:
    from config import Config
    from dataset import EndToEndDataset
    from models.crnn import EndToEndLPR
    from utils import seed_everything, decode_predictions

def test_pipeline():
    seed_everything(Config.SEED)
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET...")
    print("="*60)

    # Load Dataset
    test_path = "data/test" if os.path.exists("data/test") else Config.DATA_ROOT
    print(f"📂 Evaluating on: {test_path}")
    test_ds = EndToEndDataset(test_path, mode='test')
    
    if len(test_ds) == 0:
        print("ERROR: Test loader has no data!")
        return

    test_loader = DataLoader(
        test_ds, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False,
        collate_fn=EndToEndDataset.collate_fn, 
        num_workers=Config.NUM_WORKERS, 
        pin_memory=True
    )

    # Initialize Model
    model = EndToEndLPR(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    
    # Check for best_model.pth or use initial weights (for testing script logic)
    if os.path.exists("best_model.pth"):
        print(f"📂 Loading weights from 'best_model.pth'...")
        model.load_state_dict(torch.load("best_model.pth", map_location=Config.DEVICE, weights_only=True))
    else:
        print("WARNING: best_model.pth not found! Using random weights.")
    
    model.eval()

    test_correct = 0
    test_total = 0
    test_char_correct = 0
    test_char_total = 0

    results = [] 
    all_predictions = []

    with torch.no_grad():
        for lr_images, hr_images, targets, target_lengths, labels_text, track_ids in tqdm(test_loader, desc="Testing"):
            lr_images = lr_images.to(Config.DEVICE)
            sr_imgs, preds = model(lr_images)
            decoded = decode_predictions(torch.argmax(preds, dim=2), Config.IDX2CHAR)

            for i in range(len(labels_text)):
                gt = labels_text[i]
                pred = decoded[i]
                tid = track_ids[i]

                all_predictions.append({
                    'track_id': tid,
                    'prediction': pred,
                    'ground_truth': gt if gt else "N/A"
                })

                if gt:
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
                        results.append({'track_id': tid, 'gt': gt, 'pred': pred})

    # Save all predictions
    with open('predictions.json', 'w') as f:
        json.dump(all_predictions, f, indent=2)
    print(f"\n💾 All predictions saved to 'predictions.json'")

    if test_total > 0:
        test_acc = (test_correct / test_total) * 100
        char_acc = (test_char_correct / test_char_total) * 100 if test_char_total > 0 else 0

        print(f"\n📊 TEST RESULTS:")
        print(f"   • Exact Match Accuracy: {test_acc:.2f}% ({test_correct}/{test_total})")
        print(f"   • Character Accuracy:   {char_acc:.2f}%")

        if results:
            print(f"\n🔍 Sample Errors (first 10):")
            for i, r in enumerate(results[:10]):
                print(f"   {i+1}. [{r['track_id']}] GT: '{r['gt']}' | Pred: '{r['pred']}'")

        # Lưu kết quả vào file
        test_results = {
            'test_accuracy': test_acc,
            'char_accuracy': char_acc,
            'total_samples_with_gt': test_total,
            'correct_samples': test_correct,
            'sample_errors': results
        }
        with open('test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"💾 Metrics saved to 'test_results.json'")
    else:
        print("\nINFO: No Ground Truth labels available for accuracy calculation.")
        print(f"INFO: Completed inference for {len(all_predictions)} samples.")

if __name__ == "__main__":
    test_pipeline()
