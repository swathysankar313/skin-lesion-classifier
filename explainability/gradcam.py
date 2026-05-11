"""
explainability/gradcam.py

Grad-CAM implementation for the ViT-based skin lesion classifier.
Visualises which image regions most influenced the model's prediction.

Uses the pytorch-grad-cam library which supports Vision Transformers
via GradCAM applied to the last attention block's LayerNorm output.
"""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from models.fusion_model import MultimodalFusionClassifier
from data.augmentation import get_val_transform
from config import LESION_CLASSES, IMG_SIZE


class GradCAMExplainer:
    """
    Wraps pytorch-grad-cam for the MultimodalFusionClassifier.

    Since the model is multimodal, we supply a wrapper that passes
    metadata as a fixed context and exposes only the image input to
    Grad-CAM (which expects a single-input callable).

    Usage::

        explainer = GradCAMExplainer(model, device)
        heatmap   = explainer.explain(image_path, metadata_tensor, target_class=4)
        explainer.save(heatmap, "assets/gradcam_mel.png")
    """

    def __init__(self, model: MultimodalFusionClassifier,
                 device: torch.device):
        self.model  = model.eval().to(device)
        self.device = device

        # Target the LayerNorm after the last ViT block (best for ViT Grad-CAM)
        target_layer = [model.vit_encoder.vit.blocks[-1].norm1]

        # Wrapper: fixes metadata, exposes only image for GradCAM
        self._meta_ctx: torch.Tensor | None = None

        class _ImageOnlyWrapper(torch.nn.Module):
            def __init__(self_, inner):
                super().__init__()
                self_.inner = inner

            def forward(self_, x):
                return self_.inner(x, self._meta_ctx)

        self._wrapper = _ImageOnlyWrapper(self.model)
        self.cam = GradCAM(model=self._wrapper, target_layers=target_layer)

    # ── Core explanation ──────────────────────────────────────────────────────

    def explain(
        self,
        image_path: str,
        metadata: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap overlaid on the input image.

        Args:
            image_path:   Path to the dermoscopy image.
            metadata:     Pre-encoded metadata tensor (1, meta_dim).
            target_class: Class index to explain. If None, uses argmax.

        Returns:
            overlay: np.ndarray (H, W, 3) uint8 — heatmap overlaid on image.
        """
        # Load & transform image
        pil_img  = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        rgb_np   = np.array(pil_img, dtype=np.float32) / 255.0   # [0,1] for overlay

        transform = get_val_transform()
        tensor_img = transform(image=np.array(pil_img))["image"]
        input_tensor = tensor_img.unsqueeze(0).to(self.device)

        self._meta_ctx = metadata.to(self.device)

        # Compute Grad-CAM
        targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
        cam_map = self.cam(input_tensor=input_tensor, targets=targets)
        cam_map = cam_map[0]   # (H, W) normalised to [0,1]

        overlay = show_cam_on_image(rgb_np, cam_map, use_rgb=True)
        return overlay

    # ── Visualisation / save ──────────────────────────────────────────────────

    def visualise(self, image_path: str, metadata: torch.Tensor,
                  target_class: int | None = None):
        overlay = self.explain(image_path, metadata, target_class)
        label   = LESION_CLASSES[target_class] if target_class is not None else "top-1"

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(Image.open(image_path).resize((IMG_SIZE, IMG_SIZE)))
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(overlay)
        axes[1].set_title(f"Grad-CAM — class: {label}")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()
        return overlay

    @staticmethod
    def save(overlay: np.ndarray, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay).save(output_path)
        print(f"[GradCAM] Saved → {output_path}")