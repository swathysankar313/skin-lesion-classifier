"""
config.py — Centralised hyperparameters and paths for the
Skin Lesion Multimodal Classifier.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

METADATA_CSV   = DATA_DIR / "HAM10000_metadata.csv"
IMAGE_DIRS     = [
    DATA_DIR / "HAM10000_images_part1",
    DATA_DIR / "HAM10000_images_part2",
]
CHECKPOINT_DIR = BASE_DIR / "models" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Dataset ────────────────────────────────────────────────────────────────────
LESION_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES    = len(LESION_CLASSES)                   # 7

IMG_SIZE       = 224                                   # ViT expects 224×224
TRAIN_SPLIT    = 0.80
VAL_SPLIT      = 0.10
TEST_SPLIT     = 0.10
RANDOM_SEED    = 42

# ── Metadata features ──────────────────────────────────────────────────────────
META_CATEGORICAL = ["sex", "localization"]
META_NUMERICAL   = ["age"]
META_DIM         = 34                                  # post-encoding feature dim

# ── Model ──────────────────────────────────────────────────────────────────────
VIT_MODEL_NAME  = "vit_base_patch16_224"               # timm model ID
VIT_PRETRAINED  = True
VIT_EMBED_DIM   = 768                                  # ViT-B/16 CLS token size

META_HIDDEN_DIMS = [128, 64]                           # MetaNet hidden layers
META_OUT_DIM     = 64

FUSION_HIDDEN_DIM = 256
DROPOUT           = 0.30

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE        = 32
NUM_EPOCHS        = 30
LEARNING_RATE     = 2e-4
WEIGHT_DECAY      = 1e-4
LR_T_MAX          = 30                                 # CosineAnnealingLR period
LABEL_SMOOTHING   = 0.10
USE_AMP           = True                               # mixed precision (fp16)
NUM_WORKERS       = 4
PATIENCE          = 7                                  # early stopping

# ── Class weights (inverse frequency on HAM10000) ──────────────────────────────
# Helps with the severe class imbalance (nv >> others)
CLASS_WEIGHTS = [5.0, 5.0, 2.0, 7.0, 4.0, 1.0, 8.0]

# ── Explainability ─────────────────────────────────────────────────────────────
GRADCAM_TARGET_LAYER = "vit.blocks[-1].norm1"          # last ViT block
LIME_NUM_SAMPLES      = 1000
LIME_NUM_FEATURES     = 10