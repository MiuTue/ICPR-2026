import torch
from models.crnn import MultiFrameCRNN
from config import Config

def verify_model():
    print(f"Testing model with NUM_FRAMES: {Config.NUM_FRAMES}")
    model = MultiFrameCRNN(num_classes=Config.NUM_CLASSES)
    model.eval()
    
    # Input: [Batch, Time, Channel, H, W]
    # AdvancedMultiFrameDataset returns [B, T, C, H, W] but MultiFrameCRNN internally 
    # might expect x as [B, T, C, H, W] or [B*T, C, H, W]. 
    # Let's check MultiFrameCRNN.forward
    
    dummy_input = torch.randn(2, Config.NUM_FRAMES, 3, Config.IMG_HEIGHT, Config.IMG_WIDTH)
    
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Success! Output shape: {output.shape}")
        # Expected shape [B, T_seq, Classes] 
        # Seq length depends on Upsample and Conv layers.
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_model()
