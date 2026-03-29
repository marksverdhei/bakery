"""Additional tests for bakery.kl — per_sample mode and disable_adapters."""

import torch
from unittest.mock import MagicMock

from bakery.kl import compute_kl_divergence, disable_adapters


# ---------------------------------------------------------------------------
# compute_kl_divergence — per_sample mode
# ---------------------------------------------------------------------------

class TestKLPerSample:
    def test_per_sample_returns_batch_shaped_tensor(self):
        batch = 3
        logits = torch.randn(batch, 5, 100)
        mask = torch.ones(batch, 5)
        result = compute_kl_divergence(logits, logits, mask, per_sample=True)
        assert result.shape == (batch,)

    def test_per_sample_identical_distributions_near_zero(self):
        logits = torch.randn(2, 4, 50)
        mask = torch.ones(2, 4)
        result = compute_kl_divergence(logits, logits, mask, per_sample=True)
        assert (result < 1e-5).all()

    def test_per_sample_vs_scalar_sum_relationship(self):
        """Per-sample losses summed should be close to the scalar version
        when all samples have the same mask length."""
        teacher = torch.randn(2, 5, 50)
        student = torch.randn(2, 5, 50)
        mask = torch.ones(2, 5)
        scalar = compute_kl_divergence(teacher, student, mask, per_sample=False)
        per_sample = compute_kl_divergence(teacher, student, mask, per_sample=True)
        # scalar = mean over all 10 tokens; per_sample mean = mean of 2 per-sample means
        # They should match when both samples have equal token counts
        assert torch.isclose(per_sample.mean(), scalar, atol=1e-5)

    def test_per_sample_different_mask_lengths(self):
        """Different token counts per sample should not produce equal per-sample losses."""
        teacher = torch.randn(1, 6, 50)
        student = torch.randn(1, 6, 50)
        # mask A: all tokens; mask B: half tokens
        mask_a = torch.ones(1, 6)
        mask_b = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
        result_a = compute_kl_divergence(teacher, student, mask_a, per_sample=True)
        result_b = compute_kl_divergence(teacher, student, mask_b, per_sample=True)
        assert not torch.isclose(result_a[0], result_b[0])

    def test_per_sample_empty_mask_single_sample(self):
        """All-zero mask for a single sample should return 0 (safe clamp)."""
        logits = torch.randn(1, 4, 50)
        mask = torch.zeros(1, 4)
        result = compute_kl_divergence(logits, logits, mask, per_sample=True)
        assert result.shape == (1,)
        assert result[0].item() == pytest.approx(0.0, abs=1e-5)

    def test_per_sample_temperature_lowers_loss(self):
        teacher = torch.randn(2, 5, 100)
        student = torch.randn(2, 5, 100)
        mask = torch.ones(2, 5)
        loss_t1 = compute_kl_divergence(teacher, student, mask, temperature=1.0, per_sample=True)
        loss_t10 = compute_kl_divergence(teacher, student, mask, temperature=10.0, per_sample=True)
        assert (loss_t10 < loss_t1).all()


# ---------------------------------------------------------------------------
# disable_adapters context manager
# ---------------------------------------------------------------------------

class TestDisableAdapters:
    def test_disables_then_re_enables(self):
        model = MagicMock()
        with disable_adapters(model):
            model.disable_adapter_layers.assert_called_once()
            model.enable_adapter_layers.assert_not_called()
        model.enable_adapter_layers.assert_called_once()

    def test_re_enables_on_exception(self):
        model = MagicMock()
        try:
            with disable_adapters(model):
                raise RuntimeError("training exploded")
        except RuntimeError:
            pass
        model.disable_adapter_layers.assert_called_once()
        model.enable_adapter_layers.assert_called_once()

    def test_yields_nothing(self):
        model = MagicMock()
        with disable_adapters(model) as value:
            assert value is None


import pytest
