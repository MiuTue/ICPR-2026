import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    from .config import Config
except ImportError:
    from config import Config


def get_train_transforms(height=Config.IMG_HEIGHT, width=Config.IMG_WIDTH):
    """Get training data augmentation transforms."""
    return A.Compose([
        A.Resize(height=height, width=width),
        A.OneOf([
            A.Affine(scale=(0.8, 1.2), translate_percent=(0.1, 0.1), rotate=(-15, 15), shear=(-10, 10), p=1.0, fill=128),
            A.ElasticTransform(alpha=1, sigma=50, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
            A.OpticalDistortion(distort_limit=0.3, p=1.0),
        ], p=0.7),
        A.Perspective(scale=(0.05, 0.15), p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
        ], p=0.6),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
            A.GaussNoise(std_range=(0.1, 0.5), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.5),
        A.CoarseDropout(
            num_holes_range=(2, 8),
            hole_height_range=(4, 8),
            hole_width_range=(4, 8),
            fill=128,
            p=0.5
        ),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])


def get_degradation_transforms():
    """Get synthetic degradation transforms to simulate LR images from HR."""
    return A.Compose([
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
            A.Defocus(radius=(1, 3), alias_blur=(0.1, 0.3), p=1.0),
        ], p=0.8),
        A.OneOf([
            A.GaussNoise(std_range=(0.1, 0.5), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
        ], p=0.8),
        A.ImageCompression(quality_range=(10, 50), p=0.5),
        A.Downscale(scale_range=(0.25, 0.5), p=0.5),
    ])


def get_val_transforms(height=Config.IMG_HEIGHT, width=Config.IMG_WIDTH):
    """Get validation/inference transforms (no augmentation)."""
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
