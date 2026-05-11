"""
explainability/lime_explain.py

LIME-based explanations for the skin lesion classifier.
Perturbs image superpixels and identifies which segments most
influenced the model's prediction.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from lime import lime_image
from skimage.segmentation import mark_boundaries

from models.fusion_model import MultimodalFusionClassifier
from data.augmentation import get_val_transform
from config import LESION_CLASSES, IMG_SIZE, LIME_NUM_SAMPLES, LIME_NUM_FEATURES


class LIMEExplainer:
    """
    LIME image explainer for the MultimodalFusionClassifier.

    Metadata is held fixed during perturbation; only the image is varied,
    so LIME identifies which visual regions are important.

    Usage::

        explainer = LIMEExplainer(model, device)
        explanation = explainer.explain(image_path, metadata_tensor)
        explainer.visualise(explanation, image_path)
    """

    def __init__(self, model: MultimodalFusionClassifier,
                 device: torch.device):
        self.model  = model.eval().to(device)
        self.device = device
        self._meta_ctx: torch.Tensor | None = None
        self._explainer = lime_image.LimeImageExplainer()
        self._transform = get_val_transform()

    # ── Prediction function for LIME ──────────────────────────────────────────

    def _predict_fn(self, images: np.ndarray) -> np.ndarray:
        """
        LIME calls this with a batch of perturbed images (N, H, W, 3) uint8.
        Returns softmax probabilities (N, num_classes).
        """
        all_probs = []
        batch_size = 32

        for i in range(0, len(images), batch_size):
            batch_np = images[i:i + batch_size]
            tensors = []
            for img_np in batch_np:
                t = self._transform(image=img_np)["image"]
                tensors.append(t)
            img_tensor = torch.stack(tensors).to(self.device)

            # Repeat metadata to match batch size
            meta_batch = self._meta_ctx.repeat(img_tensor.size(0), 1)

            with torch.no_grad():
                logits = self.model(img_tensor, meta_batch)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

        return np.vstack(all_probs)

    # ── Explanation ───────────────────────────────────────────────────────────

    def explain(
        self,
        image_path: str,
        metadata: torch.Tensor,
        target_class: int | None = None,
        num_samples: int = LIME_NUM_SAMPLES,
    ):
        """
        Generate a LIME explanation.

        Args:
            image_path:   Path to dermoscopy image.
            metadata:     Pre-encoded metadata tensor (1, meta_dim).
            target_class: Class to explain. If None, uses the predicted class.
            num_samples:  Number of LIME perturbation samples.

        Returns:
            lime Explanation object
        """
        self._meta_ctx = metadata.to(self.device)

        img = np.array(Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE)))

        explanation = self._explainer.explain_instance(
            img,
            self._predict_fn,
            top_labels=7,
            hide_color=0,
            num_samples=num_samples,
        )

        if target_class is None:
            target_class = explanation.top_labels[0]

        return explanation, target_class

    # ── Visualisation ─────────────────────────────────────────────────────────

    def visualise(self, image_path: str, metadata: torch.Tensor,
                  target_class: int | None = None,
                  num_features: int = LIME_NUM_FEATURES,
                  output_path: str | None = None):
        explanation, cls = self.explain(image_path, metadata, target_class)
        label = LESION_CLASSES[cls]

        temp, mask = explanation.get_image_and_mask(
            cls,
            positive_only=True,
            num_features=num_features,
            hide_rest=False,
        )

        img_with_boundaries = mark_boundaries(temp / 255.0, mask)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(Image.open(image_path).resize((IMG_SIZE, IMG_SIZE)))
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(img_with_boundaries)
        axes[1].set_title(f"LIME Explanation — {label}")
        axes[1].axis("off")

        plt.tight_layout()

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"[LIME] Saved → {output_path}")

        plt.show()
        return img_with_boundaries