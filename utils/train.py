"""
utils/train.py

Training loop for the MultimodalFusionClassifier.

Features:
  - Mixed precision (AMP) via torch.cuda.amp
  - Early stopping on validation accuracy
  - Learning-rate scheduling (CosineAnnealingLR)
  - Two-phase training: ViT frozen (warm-up) → ViT unfrozen (fine-tune)
  - Per-epoch CSV logging + TensorBoard
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import csv, time

from models.fusion_model import MultimodalFusionClassifier
from utils.evaluate import compute_metrics
from utils.logger import Logger
from config import (
    LEARNING_RATE, WEIGHT_DECAY, LR_T_MAX,
    NUM_EPOCHS, LABEL_SMOOTHING, USE_AMP,
    CLASS_WEIGHTS, NUM_CLASSES, PATIENCE,
    CHECKPOINT_DIR, LESION_CLASSES,
)


def build_optimizer_and_scheduler(model: MultimodalFusionClassifier, lr: float = LEARNING_RATE):
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=LR_T_MAX, eta_min=1e-6)
    return optimizer, scheduler


def train(
    model: MultimodalFusionClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = NUM_EPOCHS,
    warmup_epochs: int = 5,
    device: torch.device | None = None,
) -> MultimodalFusionClassifier:
    """
    Full training loop with warm-up + fine-tuning phases.

    Phase 1 (epochs 1 – warmup_epochs):   ViT frozen, only head + MetaNet trained.
    Phase 2 (epochs warmup_epochs+1 – N): Last 4 ViT blocks unfrozen.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)

    optimizer, scheduler = build_optimizer_and_scheduler(model)
    scaler = GradScaler(enabled=USE_AMP and device.type == "cuda")
    logger = Logger()

    best_val_acc = 0.0
    patience_counter = 0
    best_ckpt = CHECKPOINT_DIR / "best_model.pth"

    print(f"\n[Train] Starting on {device}  |  Epochs: {num_epochs}  "
          f"|  Warm-up: {warmup_epochs}\n")

    for epoch in range(1, num_epochs + 1):
        # ── Phase transition ──────────────────────────────────────────────────
        if epoch == warmup_epochs + 1:
            print("\n[Train] Warm-up complete — unfreezing last 4 ViT blocks.\n")
            model.unfreeze_vit(n_blocks=4)
            # Rebuild optimizer to include new parameters
            optimizer, scheduler = build_optimizer_and_scheduler(
                model, lr=LEARNING_RATE * 0.1
            )

        # ── Training pass ─────────────────────────────────────────────────────
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        t0 = time.time()

        for images, metadata, labels in tqdm(train_loader,
                                             desc=f"Epoch {epoch}/{num_epochs} [train]",
                                             leave=False):
            images   = images.to(device)
            metadata = metadata.to(device)
            labels   = labels.to(device)

            optimizer.zero_grad()
            with autocast(enabled=USE_AMP and device.type == "cuda"):
                logits = model(images, metadata)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * labels.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        scheduler.step()
        avg_train_loss = train_loss / total
        train_acc      = correct / total

        # ── Validation pass ───────────────────────────────────────────────────
        val_metrics = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}  Acc: {val_metrics['acc']:.4f}  "
            f"F1: {val_metrics['f1']:.4f} | "
            f"{elapsed:.1f}s"
        )
        logger.log(epoch, avg_train_loss, train_acc,
                   val_metrics["loss"], val_metrics["acc"], val_metrics["f1"])

        # ── Early stopping & checkpoint ───────────────────────────────────────
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            model.save(best_ckpt)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n[Train] Early stopping at epoch {epoch}. "
                      f"Best val acc: {best_val_acc:.4f}")
                break

    print(f"\n[Train] Done. Best validation accuracy: {best_val_acc:.4f}")
    model.load(best_ckpt, device=device)
    return model


def evaluate(
    model: MultimodalFusionClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Single evaluation pass. Returns dict with loss, acc, f1."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, metadata, labels in loader:
            images   = images.to(device)
            metadata = metadata.to(device)
            labels   = labels.to(device)

            with autocast(enabled=USE_AMP and device.type == "cuda"):
                logits = model(images, metadata)
                loss   = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / total
    metrics["acc"]  = correct / total
    return metrics