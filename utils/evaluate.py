"""
utils/evaluate.py

Evaluation utilities: F1-score, AUC, confusion matrix, per-class report.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, roc_auc_score,
)
from torch.utils.data import DataLoader
from pathlib import Path

from config import LESION_CLASSES


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """
    Returns macro F1 and per-class accuracy.
    AUC requires probabilities — computed separately in full_evaluation().
    """
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"f1": f1}


def full_evaluation(model, loader: DataLoader,
                    device: torch.device,
                    output_dir: str = "assets/") -> dict:
    """
    Full evaluation pass on loader.
    Saves confusion matrix and per-class report to output_dir.

    Returns dict with acc, f1, auc (macro OvR).
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, metadata, labels in loader:
            images   = images.to(device)
            metadata = metadata.to(device)
            logits   = model(images, metadata)
            probs    = torch.softmax(logits, dim=1).cpu().numpy()
            preds    = logits.argmax(1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.append(probs)

    all_probs = np.vstack(all_probs)
    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    # ── Classification report ─────────────────────────────────────────────────
    report = classification_report(
        all_labels, all_preds,
        target_names=LESION_CLASSES, zero_division=0,
    )
    print("\n" + "═" * 60)
    print("Classification Report")
    print("═" * 60)
    print(report)
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {f1:.4f}")
    print(f"Macro AUC: {auc:.4f}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    _plot_confusion_matrix(cm, output_dir)

    return {"acc": acc, "f1": f1, "auc": auc}


def _plot_confusion_matrix(cm: np.ndarray, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LESION_CLASSES,
        yticklabels=LESION_CLASSES, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix — Skin Lesion Classifier", fontsize=14)
    plt.tight_layout()
    out_path = Path(output_dir) / "confusion_matrix.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Evaluate] Confusion matrix saved → {out_path}")