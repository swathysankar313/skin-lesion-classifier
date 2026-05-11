"""
models/vit_model.py

Vision Transformer (ViT-B/16) image encoder loaded from timm.
Returns the [CLS] token embedding (768-d) for use in the fusion model.
"""

import torch
import torch.nn as nn
import timm
from config import VIT_MODEL_NAME, VIT_PRETRAINED, VIT_EMBED_DIM


class ViTEncoder(nn.Module):
    """
    Wraps a pretrained ViT-B/16 from timm, replacing the classification
    head with an identity so that the forward pass returns the CLS token
    embedding (shape: [B, 768]).

    The backbone is optionally frozen for the first few epochs (feature
    extraction), then unfrozen for fine-tuning.

    Args:
        model_name:  timm model identifier (default: vit_base_patch16_224)
        pretrained:  Load ImageNet pretrained weights
        freeze:      If True, freeze all backbone weights initially
    """

    def __init__(
        self,
        model_name: str = VIT_MODEL_NAME,
        pretrained: bool = VIT_PRETRAINED,
        freeze: bool = False,
    ):
        super().__init__()
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,          # remove classification head → returns CLS embedding
        )
        self.embed_dim = self.vit.embed_dim   # 768 for ViT-B/16

        if freeze:
            self.freeze_backbone()

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: image tensor of shape (B, 3, 224, 224)
        Returns:
            CLS embedding of shape (B, 768)
        """
        return self.vit(x)

    # ── Freeze / unfreeze ─────────────────────────────────────────────────────

    def freeze_backbone(self):
        """Freeze all ViT parameters (for warm-up / feature extraction phase)."""
        for param in self.vit.parameters():
            param.requires_grad = False
        print("[ViTEncoder] Backbone frozen.")

    def unfreeze_backbone(self, unfreeze_last_n_blocks: int = 4):
        """
        Unfreeze the last *n* transformer blocks for fine-tuning.
        Also unfreezes the patch embedding and norm layers.
        """
        for param in self.vit.parameters():
            param.requires_grad = False

        # Unfreeze last n blocks
        blocks = list(self.vit.blocks)
        for block in blocks[-unfreeze_last_n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

        # Unfreeze final norm
        for param in self.vit.norm.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.vit.parameters() if p.requires_grad)
        print(f"[ViTEncoder] Unfrozen last {unfreeze_last_n_blocks} blocks "
              f"({trainable:,} trainable params).")

    def unfreeze_all(self):
        for param in self.vit.parameters():
            param.requires_grad = True
        print("[ViTEncoder] All backbone params unfrozen.")