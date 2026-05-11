"""
tests/test_models.py

Unit tests for model components.
Validates shapes, dtypes, and forward pass consistency without
requiring the full HAM10000 dataset.
"""

import pytest
import torch
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.meta_net import MetaNet
from models.fusion_model import MultimodalFusionClassifier
from config import NUM_CLASSES, META_OUT_DIM, VIT_EMBED_DIM


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="module")
def meta_net():
    return MetaNet(input_dim=34, hidden_dims=[64, 32], output_dim=32)


@pytest.fixture(scope="module")
def fusion_model():
    """Small fusion model (ViT still loaded from timm — uses pretrained=False for speed)."""
    import timm
    model = MultimodalFusionClassifier(freeze_vit=True)
    return model


@pytest.fixture
def dummy_image():
    return torch.randn(2, 3, 224, 224)


@pytest.fixture
def dummy_meta():
    return torch.randn(2, 34)


# ── MetaNet tests ─────────────────────────────────────────────────────────────

class TestMetaNet:
    def test_output_shape(self, meta_net, dummy_meta):
        meta_net.eval()
        with torch.no_grad():
            out = meta_net(dummy_meta[:, :34])
        assert out.shape == (2, 32), f"Expected (2,32) got {out.shape}"

    def test_output_dtype(self, meta_net, dummy_meta):
        meta_net.eval()
        with torch.no_grad():
            out = meta_net(dummy_meta[:, :34])
        assert out.dtype == torch.float32

    def test_no_nan_output(self, meta_net, dummy_meta):
        meta_net.eval()
        with torch.no_grad():
            out = meta_net(dummy_meta[:, :34])
        assert not torch.isnan(out).any()

    def test_batch_size_1(self, meta_net):
        """Edge case: batch of 1 (BatchNorm should handle this in eval mode)."""
        meta_net.eval()
        x = torch.randn(1, 34)
        with torch.no_grad():
            out = meta_net(x)
        assert out.shape == (1, 32)


# ── FusionModel tests ─────────────────────────────────────────────────────────

class TestFusionModel:
    def test_forward_output_shape(self, fusion_model, dummy_image, dummy_meta):
        fusion_model.eval()
        with torch.no_grad():
            logits = fusion_model(dummy_image, dummy_meta)
        assert logits.shape == (2, NUM_CLASSES), \
            f"Expected (2, {NUM_CLASSES}) got {logits.shape}"

    def test_predict_proba_sums_to_one(self, fusion_model, dummy_image, dummy_meta):
        fusion_model.eval()
        probs = fusion_model.predict_proba(dummy_image, dummy_meta)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5), \
            f"Probs don't sum to 1: {sums}"

    def test_predict_class_in_range(self, fusion_model, dummy_image, dummy_meta):
        fusion_model.eval()
        classes = fusion_model.predict_class(dummy_image, dummy_meta)
        assert classes.shape == (2,)
        assert all(0 <= c < NUM_CLASSES for c in classes.tolist())

    def test_no_nan_in_logits(self, fusion_model, dummy_image, dummy_meta):
        fusion_model.eval()
        with torch.no_grad():
            logits = fusion_model(dummy_image, dummy_meta)
        assert not torch.isnan(logits).any(), "NaN found in logits"

    def test_checkpoint_save_load(self, fusion_model, dummy_image, dummy_meta, tmp_path):
        """Save and reload checkpoint, verify identical outputs."""
        ckpt = tmp_path / "test_ckpt.pth"
        fusion_model.eval()
        with torch.no_grad():
            logits_before = fusion_model(dummy_image, dummy_meta)
        fusion_model.save(ckpt)

        model2 = MultimodalFusionClassifier(freeze_vit=True)
        model2.load(ckpt)
        model2.eval()
        with torch.no_grad():
            logits_after = model2(dummy_image, dummy_meta)

        assert torch.allclose(logits_before, logits_after, atol=1e-5), \
            "Outputs differ after save/load"


# ── Augmentation tests ────────────────────────────────────────────────────────

class TestAugmentations:
    def test_train_transform_shape(self):
        from data.augmentation import get_train_transform
        import numpy as np
        t = get_train_transform()
        img = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        result = t(image=img)["image"]
        assert result.shape == (3, 224, 224), f"Got {result.shape}"

    def test_val_transform_deterministic(self):
        from data.augmentation import get_val_transform
        import numpy as np
        t = get_val_transform()
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        out1 = t(image=img)["image"]
        out2 = t(image=img)["image"]
        assert torch.equal(out1, out2), "Val transform should be deterministic"