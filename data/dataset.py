"""
data/dataset.py

PyTorch Dataset for the HAM10000 skin lesion benchmark.
Each sample returns (image_tensor, metadata_tensor, label).
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Callable

from config import LESION_CLASSES, IMAGE_DIRS, NUM_CLASSES


class HAM10000Dataset(Dataset):
    """
    HAM10000 multimodal dataset.

    Args:
        df:           DataFrame slice (train / val / test).
        meta_array:   Pre-encoded metadata array, shape (N, meta_dim).
        transform:    Albumentations transform pipeline.
        image_dirs:   List of directories to search for image files.

    Returns per sample:
        image    — FloatTensor (3, 224, 224)
        metadata — FloatTensor (meta_dim,)
        label    — LongTensor  scalar
    """

    CLASS_TO_IDX = {cls: i for i, cls in enumerate(LESION_CLASSES)}

    def __init__(
        self,
        df: pd.DataFrame,
        meta_array: np.ndarray,
        transform: Optional[Callable] = None,
        image_dirs: list[str | Path] = IMAGE_DIRS,
    ):
        assert len(df) == len(meta_array), "df and meta_array must have the same length."
        self.df         = df.reset_index(drop=True)
        self.meta       = torch.tensor(meta_array, dtype=torch.float32)
        self.transform  = transform
        self.image_dirs = [Path(d) for d in image_dirs]

        # Build image_id → file path lookup once at construction
        self._path_cache = self._build_path_cache()

    # ── Path lookup ───────────────────────────────────────────────────────────

    def _build_path_cache(self) -> dict[str, Path]:
        cache: dict[str, Path] = {}
        for d in self.image_dirs:
            if not d.exists():
                continue
            for fp in d.glob("*.jpg"):
                cache[fp.stem] = fp          # stem == image_id
        return cache

    def _get_image_path(self, image_id: str) -> Path:
        if image_id in self._path_cache:
            return self._path_cache[image_id]
        raise FileNotFoundError(
            f"Image '{image_id}.jpg' not found in any of {self.image_dirs}"
        )

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row      = self.df.iloc[idx]
        img_path = self._get_image_path(row["image_id"])

        image = np.array(Image.open(img_path).convert("RGB"))
        if self.transform:
            image = self.transform(image=image)["image"]   # albumentations API

        label = self.CLASS_TO_IDX[row["dx"]]
        return image, self.meta[idx], torch.tensor(label, dtype=torch.long)

    # ── Class distribution ────────────────────────────────────────────────────

    def class_distribution(self) -> dict[str, int]:
        return self.df["dx"].value_counts().to_dict()


# ── DataLoader factory ────────────────────────────────────────────────────────

def build_dataloaders(
    df_train, df_val, df_test,
    meta_train, meta_val, meta_test,
    train_transform, val_transform,
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Returns (train_loader, val_loader, test_loader)."""

    train_ds = HAM10000Dataset(df_train, meta_train, transform=train_transform)
    val_ds   = HAM10000Dataset(df_val,   meta_val,   transform=val_transform)
    test_ds  = HAM10000Dataset(df_test,  meta_test,  transform=val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader