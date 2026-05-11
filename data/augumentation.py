"""
data/augmentation.py

Albumentations-based augmentation pipelines for HAM10000 dermoscopy images.

Training pipeline uses aggressive transforms to counteract overfitting
on a ~8k image training set. Validation/test use only resize + normalise.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import IMG_SIZE

# ImageNet statistics (ViT pretrained on ImageNet)
_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)


def get_train_transform() -> A.Compose:
    """
    Augmentation strategy for training:
      - Spatial: flip, rotate, elastic, grid distortion, crop
      - Colour: brightness/contrast jitter, hue-saturation, CLAHE
      - Regularisation: coarse dropout (cutout)
      - Dermoscopy-specific: hair-like artefacts via CoarseDropout lines
    """
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),

        # ── Spatial transforms ────────────────────────────────────────────────
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=30, p=0.6),
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),
        A.GridDistortion(p=0.2),
        A.RandomResizedCrop(height=IMG_SIZE, width=IMG_SIZE,
                            scale=(0.80, 1.0), p=0.4),

        # ── Colour / photometric transforms ───────────────────────────────────
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10,
                             sat_shift_limit=20,
                             val_shift_limit=10, p=0.4),
        A.CLAHE(clip_limit=2.0, p=0.3),              # enhances dermoscopy contrast
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),

        # ── Regularisation ────────────────────────────────────────────────────
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16,
                        fill_value=0, p=0.3),

        # ── Normalise & tensor ────────────────────────────────────────────────
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])


def get_val_transform() -> A.Compose:
    """Deterministic: resize + normalise only."""
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])


# Convenience alias
get_test_transform = get_val_transform