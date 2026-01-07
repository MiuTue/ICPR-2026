import torch
from models.crnn import MultiFrameCRNN

def test_forward():
    num_classes = 100
    batch_size = 2
    num_frames = 5
    channels = 3
    height = 32
    width = 128
    
    model = MultiFrameCRNN(num_classes=num_classes)
    images = torch.randn(batch_size, num_frames, channels, height, width)
    
    try:
        preds = model(images)
        print(f"Success! Output shape: {preds.shape}")
    except Exception as e:
        print(f"Failed! Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_forward()
