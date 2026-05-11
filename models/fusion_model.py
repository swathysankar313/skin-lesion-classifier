"""
models/fusion_model.py

Multimodal Fusion Classifier — the top-level model that combines
the ViT image encoder and MetaNet metadata encoder via concatenation,
then passes the joint representation through a classification head.
"""

import torch
import torch.nn as nn
from pathlib import Path

from models.vit_model import ViTEncoder
from models.meta_net import MetaNet
from config import (
    VIT_EMBED_DIM, META_OUT_DIM,
    FUSION_HIDDEN_DIM, DROPOUT, NUM_CLASSES,
)


class MultimodalFusionClassifier(nn.Module):
    """
    Architecture::

        image  → ViTEncoder  → (B, 768)  ─┐
                                            concat → (B, 832)
        meta   → MetaNet     → (B,  64)  ─┘
                                            FC(832→256) → ReLU → Dropout
                                            FC(256→7)   → logits

    Args:
        num_classes:      Number of output classes (7 for HAM10000).
        fusion_hidden:    Hidden size of the fusion MLP.
        dropout:          Dropout rate in the fusion head.
        freeze_vit:       Freeze ViT backbone on init (use for warm-up).
        meta_input_dim:   Dimensionality of preprocessed metadata.
    """

    def __init__(
        self,
        num_classes:    int   = NUM_CLASSES,
        fusion_hidden:  int   = FUSION_HIDDEN_DIM,
        dropout:        float = DROPOUT,
        freeze_vit:     bool  = True,
        meta_input_dim: int   = META_OUT_DIM,
    ):
        super().__init__()

        self.vit_encoder = ViTEncoder(freeze=freeze_vit)
        self.meta_encoder = MetaNet(output_dim=META_OUT_DIM)

        combined_dim = VIT_EMBED_DIM + META_OUT_DIM    # 768 + 64 = 832

        self.fusion_head = nn.Sequential(
            nn.Linear(combined_dim, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes),
        )

        self._init_fusion_weights()

    # ── Weight init ───────────────────────────────────────────────────────────

    def _init_fusion_weights(self):
        for m in self.fusion_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image:    (B, 3, 224, 224)
            metadata: (B, meta_dim)

        Returns:
            logits:   (B, num_classes)  — raw unnormalised scores
        """
        img_feat  = self.vit_encoder(image)       # (B, 768)
        meta_feat = self.meta_encoder(metadata)   # (B, 64)
        fused     = torch.cat([img_feat, meta_feat], dim=1)   # (B, 832)
        return self.fusion_head(fused)            # (B, 7)

    # ── Convenience methods ───────────────────────────────────────────────────

    def predict_proba(self, image: torch.Tensor,
                      metadata: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        with torch.no_grad():
            logits = self.forward(image, metadata)
        return torch.softmax(logits, dim=1)

    def predict_class(self, image: torch.Tensor,
                      metadata: torch.Tensor) -> torch.Tensor:
        """Return argmax class indices."""
        return self.predict_proba(image, metadata).argmax(dim=1)

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def save(self, path: str | Path):
        torch.save(self.state_dict(), str(path))
        print(f"[Model] Saved checkpoint → {path}")

    def load(self, path: str | Path, device: torch.device | None = None):
        device = device or torch.device("cpu")
        state = torch.load(str(path), map_location=device)
        self.load_state_dict(state)
        print(f"[Model] Loaded checkpoint ← {path}")
        return self

    # ── Unfreeze schedule ─────────────────────────────────────────────────────

    def unfreeze_vit(self, n_blocks: int = 4):
        """Call after warm-up epochs to fine-tune last n ViT blocks."""
        self.vit_encoder.unfreeze_backbone(n_blocks)

    def count_parameters(self) -> int:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] Total params: {total:,}  |  Trainable: {trainable:,}")
        return trainable