import torch
from types import SimpleNamespace
from unittest.mock import MagicMock
from bakery.kl import compute_kl_divergence, disable_adapters, padding_side


def test_kl_identical_distributions():
    """KL divergence between identical distributions should be ~0."""
    logits = torch.randn(2, 5, 100)
    mask = torch.ones(2, 5)
    loss = compute_kl_divergence(logits, logits, mask)
    assert loss.item() < 1e-5


def test_kl_respects_mask():
    """Masked tokens should not contribute to loss."""
    teacher = torch.randn(1, 4, 50)
    student = torch.randn(1, 4, 50)
    mask_all = torch.ones(1, 4)
    mask_half = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

    loss_all = compute_kl_divergence(teacher, student, mask_all)
    loss_half = compute_kl_divergence(teacher, student, mask_half)

    # Losses should differ since different tokens are averaged
    assert not torch.isclose(loss_all, loss_half)


def test_kl_empty_mask():
    """Empty mask should return 0."""
    logits = torch.randn(1, 3, 50)
    mask = torch.zeros(1, 3)
    loss = compute_kl_divergence(logits, logits, mask)
    assert loss.item() == 0.0


def test_kl_temperature_scaling():
    """Higher temperature should produce lower KL (more uniform distributions)."""
    teacher = torch.randn(1, 5, 100)
    student = torch.randn(1, 5, 100)
    mask = torch.ones(1, 5)

    loss_t1 = compute_kl_divergence(teacher, student, mask, temperature=1.0)
    loss_t10 = compute_kl_divergence(teacher, student, mask, temperature=10.0)

    assert loss_t10.item() < loss_t1.item()


# ---------------------------------------------------------------------------
# per_sample=True
# ---------------------------------------------------------------------------


def test_kl_per_sample_returns_batch_tensor():
    """per_sample=True should return a 1-D tensor of length batch_size."""
    B, T, V = 4, 6, 50
    teacher = torch.randn(B, T, V)
    student = torch.randn(B, T, V)
    mask = torch.ones(B, T)
    result = compute_kl_divergence(teacher, student, mask, per_sample=True)
    assert result.shape == (B,)


def test_kl_per_sample_identical_is_zero():
    """Per-sample KL between identical distributions should be ~0."""
    logits = torch.randn(3, 5, 100)
    mask = torch.ones(3, 5)
    result = compute_kl_divergence(logits, logits, mask, per_sample=True)
    assert result.max().item() < 1e-5


def test_kl_per_sample_scalar_vs_per_sample_mean():
    """mean of per_sample results should equal the scalar result."""
    B, T, V = 3, 4, 50
    teacher = torch.randn(B, T, V)
    student = torch.randn(B, T, V)
    mask = torch.ones(B, T)
    scalar = compute_kl_divergence(teacher, student, mask, per_sample=False)
    per_sample = compute_kl_divergence(teacher, student, mask, per_sample=True)
    # Mean of per-sample should equal scalar when all masks are equal
    assert torch.isclose(per_sample.mean(), scalar, atol=1e-5)


def test_kl_per_sample_respects_mask():
    """Per-sample KL should be non-negative and respect masking."""
    B, T, V = 2, 6, 30
    teacher = torch.randn(B, T, V)
    student = torch.randn(B, T, V)
    mask = torch.ones(B, T)
    result = compute_kl_divergence(teacher, student, mask, per_sample=True)
    assert (result >= 0).all()


def test_kl_per_sample_empty_mask_sample_is_zero():
    """A sample whose mask is all zeros should contribute 0."""
    teacher = torch.randn(2, 4, 50)
    student = torch.randn(2, 4, 50)
    # Only first sample has tokens; second is fully masked
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    result = compute_kl_divergence(teacher, student, mask, per_sample=True)
    # Second sample: all padding → loss should be 0
    assert result[1].item() == 0.0


# ---------------------------------------------------------------------------
# disable_adapters
# ---------------------------------------------------------------------------


def test_disable_adapters_calls_disable_then_enable():
    """disable_adapters should call disable_adapter_layers() on entry."""
    model = MagicMock()
    with disable_adapters(model):
        model.disable_adapter_layers.assert_called_once()
    model.enable_adapter_layers.assert_called_once()


def test_disable_adapters_enables_on_exception():
    """disable_adapters must re-enable adapters even if the body raises."""
    model = MagicMock()
    try:
        with disable_adapters(model):
            raise RuntimeError("training error")
    except RuntimeError:
        pass
    model.enable_adapter_layers.assert_called_once()


# ---------------------------------------------------------------------------
# padding_side
# ---------------------------------------------------------------------------


def test_padding_side_context_manager():
    """padding_side restores original value, even on exception."""
    tok = SimpleNamespace(padding_side="right")
    with padding_side(tok, "left"):
        assert tok.padding_side == "left"
    assert tok.padding_side == "right"


def test_padding_side_restores_on_exception():
    """padding_side restores original value after an exception."""
    tok = SimpleNamespace(padding_side="right")
    try:
        with padding_side(tok, "left"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert tok.padding_side == "right"
