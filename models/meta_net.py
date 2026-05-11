"""
models/meta_net.py

Metadata fully-connected network (MetaNet).
Encodes patient metadata (age, sex, lesion location) into a dense
feature vector for fusion with the ViT image embedding.
"""

import torch
import torch.nn as nn
from typing import list as List
from config import META_DIM, META_HIDDEN_DIMS, META_OUT_DIM, DROPOUT


class MetaNet(nn.Module):
    """
    Small MLP that processes encoded patient metadata.

    Architecture::

        Linear(meta_dim → 128) → BN → ReLU → Dropout
        Linear(128 → 64)        → BN → ReLU → Dropout

    Args:
        input_dim:   Dimensionality of the preprocessed metadata vector.
        hidden_dims: List of hidden layer sizes.
        output_dim:  Size of the final metadata embedding.
        dropout:     Dropout probability applied after each activation.
    """

    def __init__(
        self,
        input_dim:   int       = META_DIM,
        hidden_dims: list[int] = META_HIDDEN_DIMS,
        output_dim:  int       = META_OUT_DIM,
        dropout:     float     = DROPOUT,
    ):
        super().__init__()

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev_dim = h

        layers += [
            nn.Linear(prev_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        ]

        self.net = nn.Sequential(*layers)
        self.output_dim = output_dim
        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: metadata tensor of shape (B, input_dim)
        Returns:
            embedding of shape (B, output_dim)
        """
        return self.net(x)