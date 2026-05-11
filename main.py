"""
main.py — CLI for the Skin Lesion Multimodal Classifier.

Modes:
    train     — Preprocess data, train model, save best checkpoint.
    evaluate  — Load checkpoint and evaluate on val/test set.
    explain   — Generate Grad-CAM or LIME explanation for a single image.

Usage:
    python main.py --mode train
    python main.py --mode evaluate --checkpoint models/checkpoints/best_model.pth
    python main.py --mode explain --image path/to/img.jpg --method gradcam
    python main.py --mode explain --image path/to/img.jpg --method lime
"""

import argparse
import torch
from pathlib import Path


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Using device: {device}")
    return device


# ── Train ─────────────────────────────────────────────────────────────────────

def mode_train():
    from data.preprocess import load_and_split_metadata, MetadataPreprocessor
    from data.augmentation import get_train_transform, get_val_transform
    from data.dataset import build_dataloaders
    from models.fusion_model import MultimodalFusionClassifier
    from utils.train import train
    from config import BATCH_SIZE, NUM_WORKERS

    device = get_device()

    print("[Main] Loading and splitting metadata …")
    df_train, df_val, df_test = load_and_split_metadata()

    print("[Main] Fitting metadata preprocessor …")
    preprocessor = MetadataPreprocessor()
    meta_train = preprocessor.fit_transform(df_train)
    meta_val   = preprocessor.transform(df_val)
    meta_test  = preprocessor.transform(df_test)
    preprocessor.save("data/preprocessor.pkl")

    print("[Main] Building dataloaders …")
    train_loader, val_loader, test_loader = build_dataloaders(
        df_train, df_val, df_test,
        meta_train, meta_val, meta_test,
        get_train_transform(), get_val_transform(),
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
    )

    print("[Main] Initialising model …")
    model = MultimodalFusionClassifier(freeze_vit=True)
    model.count_parameters()

    print("[Main] Starting training …")
    model = train(model, train_loader, val_loader, device=device)

    print("[Main] Final evaluation on test set …")
    from utils.evaluate import full_evaluation
    full_evaluation(model, test_loader, device)


# ── Evaluate ──────────────────────────────────────────────────────────────────

def mode_evaluate(checkpoint: str):
    from data.preprocess import load_and_split_metadata, MetadataPreprocessor
    from data.augmentation import get_val_transform
    from data.dataset import build_dataloaders, HAM10000Dataset
    from torch.utils.data import DataLoader
    from models.fusion_model import MultimodalFusionClassifier
    from utils.evaluate import full_evaluation
    from config import BATCH_SIZE, NUM_WORKERS

    device = get_device()

    df_train, df_val, df_test = load_and_split_metadata()
    preprocessor = MetadataPreprocessor.load("data/preprocessor.pkl")
    meta_val  = preprocessor.transform(df_val)
    meta_test = preprocessor.transform(df_test)

    _, val_loader, test_loader = build_dataloaders(
        df_train, df_val, df_test,
        preprocessor.transform(df_train), meta_val, meta_test,
        get_val_transform(), get_val_transform(),
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
    )

    model = MultimodalFusionClassifier()
    model.load(checkpoint, device=device)
    model.to(device)

    print("\n[Main] Validation set:")
    full_evaluation(model, val_loader, device, output_dir="assets/val/")

    print("\n[Main] Test set:")
    full_evaluation(model, test_loader, device, output_dir="assets/test/")


# ── Explain ───────────────────────────────────────────────────────────────────

def mode_explain(image_path: str, method: str, checkpoint: str,
                 target_class: int | None = None):
    import numpy as np
    from data.preprocess import MetadataPreprocessor
    import pandas as pd
    from models.fusion_model import MultimodalFusionClassifier

    device = get_device()

    model = MultimodalFusionClassifier()
    model.load(checkpoint, device=device)
    model.eval().to(device)

    # Dummy metadata (replace with real values for a specific patient)
    preprocessor = MetadataPreprocessor.load("data/preprocessor.pkl")
    dummy_row = pd.DataFrame([{"age": 45, "sex": "male", "localization": "back"}])
    meta_tensor = torch.tensor(preprocessor.transform(dummy_row),
                               dtype=torch.float32).unsqueeze(0)

    if method == "gradcam":
        from explainability.gradcam import GradCAMExplainer
        explainer = GradCAMExplainer(model, device)
        overlay = explainer.visualise(image_path, meta_tensor, target_class)
        explainer.save(overlay, "assets/gradcam_output.png")

    elif method == "lime":
        from explainability.lime_explain import LIMEExplainer
        explainer = LIMEExplainer(model, device)
        explainer.visualise(image_path, meta_tensor, target_class,
                            output_path="assets/lime_output.png")
    else:
        print(f"[Main] Unknown method: {method}. Use 'gradcam' or 'lime'.")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Skin Lesion Multimodal Classifier"
    )
    p.add_argument("--mode", choices=["train", "evaluate", "explain"],
                   required=True)
    p.add_argument("--checkpoint",
                   default="models/checkpoints/best_model.pth",
                   help="Path to model checkpoint (.pth)")
    p.add_argument("--image",   default=None, help="Image path for explain mode")
    p.add_argument("--method",  default="gradcam", choices=["gradcam", "lime"])
    p.add_argument("--target",  default=None, type=int,
                   help="Target class index for explanation (optional)")
    return p.parse_args()


if __name__ == "__main__":
    import torch
    args = parse_args()

    if args.mode == "train":
        mode_train()
    elif args.mode == "evaluate":
        mode_evaluate(args.checkpoint)
    elif args.mode == "explain":
        if not args.image:
            print("[Main] --image is required for explain mode.")
        else:
            mode_explain(args.image, args.method, args.checkpoint, args.target)