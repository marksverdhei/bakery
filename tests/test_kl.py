import torch
from types import SimpleNamespace
from bakery.kl import compute_kl_divergence, padding_side


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
