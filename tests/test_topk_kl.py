"""Tests for top-k forward KL — the sparse-friendly distillation loss."""

import math

import torch

from bakery.kl import compute_kl_divergence, topk_forward_kl
from bakery.teachers.base import topk_from_logits


# ---------- topk_from_logits ----------


def test_topk_from_logits_returns_renormalized_logprobs():
    logits = torch.randn(2, 3, 100)
    idx, lp = topk_from_logits(logits, top_k=10)
    assert idx.shape == (2, 3, 10)
    assert lp.shape == (2, 3, 10)
    # Renormalized: exp sums to 1 along last dim.
    sums = lp.exp().sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_topk_from_logits_k_capped_at_vocab():
    logits = torch.randn(1, 1, 5)
    idx, lp = topk_from_logits(logits, top_k=10)
    assert idx.shape[-1] == 5


def test_topk_from_logits_picks_actual_topk():
    logits = torch.zeros(1, 1, 5)
    logits[0, 0, 2] = 10.0
    logits[0, 0, 4] = 5.0
    idx, lp = topk_from_logits(logits, top_k=2)
    assert set(idx[0, 0].tolist()) == {2, 4}


# ---------- topk_forward_kl ----------


def test_topk_kl_zero_when_distributions_match():
    """If teacher == student logits, top-k KL should be ~0."""
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 50, requires_grad=True)
    idx, lp = topk_from_logits(logits.detach(), top_k=20)
    mask = torch.ones(1, 4)
    loss = topk_forward_kl(
        student_logits=logits,
        teacher_topk_indices=idx,
        teacher_topk_logprobs=lp,
        mask=mask,
    )
    assert loss.item() < 1e-4


def test_topk_kl_dense_matches_full_kl():
    """When K = vocab, top-k KL should equal full-vocab forward KL."""
    torch.manual_seed(0)
    V = 32
    teacher_logits = torch.randn(2, 5, V)
    student_logits = torch.randn(2, 5, V)
    mask = torch.ones(2, 5)

    full = compute_kl_divergence(teacher_logits, student_logits, mask)

    idx, lp = topk_from_logits(teacher_logits, top_k=V)
    sparse = topk_forward_kl(
        student_logits=student_logits,
        teacher_topk_indices=idx,
        teacher_topk_logprobs=lp,
        mask=mask,
    )
    assert torch.allclose(full, sparse, atol=1e-4)


def test_topk_kl_per_sample_shape():
    torch.manual_seed(1)
    logits = torch.randn(3, 4, 20)
    idx, lp = topk_from_logits(logits, top_k=8)
    mask = torch.ones(3, 4)
    out = topk_forward_kl(
        student_logits=torch.randn(3, 4, 20),
        teacher_topk_indices=idx,
        teacher_topk_logprobs=lp,
        mask=mask,
        per_sample=True,
    )
    assert out.shape == (3,)


def test_topk_kl_mask_zero_returns_zero():
    logits = torch.randn(1, 3, 20)
    idx, lp = topk_from_logits(logits, top_k=4)
    mask = torch.zeros(1, 3)
    out = topk_forward_kl(
        student_logits=torch.randn(1, 3, 20),
        teacher_topk_indices=idx,
        teacher_topk_logprobs=lp,
        mask=mask,
    )
    assert out.item() == 0.0


def test_topk_kl_differentiable_through_student():
    torch.manual_seed(2)
    student_logits = torch.randn(1, 3, 20, requires_grad=True)
    teacher_logits = torch.randn(1, 3, 20)
    idx, lp = topk_from_logits(teacher_logits, top_k=5)
    mask = torch.ones(1, 3)
    loss = topk_forward_kl(student_logits, idx, lp, mask)
    loss.backward()
    assert student_logits.grad is not None
    assert student_logits.grad.abs().sum() > 0


def test_topk_kl_temperature_scales_loss():
    """Higher temperature softens distributions → smaller KL."""
    torch.manual_seed(3)
    V = 50
    teacher_logits = torch.randn(1, 4, V) * 3.0
    student_logits = torch.randn(1, 4, V) * 3.0
    idx, lp_t1 = topk_from_logits(teacher_logits, top_k=20)
    idx_t2, lp_t2 = topk_from_logits(teacher_logits / 2.0, top_k=20)
    mask = torch.ones(1, 4)
    loss_t1 = topk_forward_kl(student_logits, idx, lp_t1, mask, temperature=1.0)
    loss_t2 = topk_forward_kl(student_logits, idx_t2, lp_t2, mask, temperature=2.0)
    # T=2 should produce a different (typically smaller for sharp distros) loss.
    assert not math.isclose(loss_t1.item(), loss_t2.item(), abs_tol=1e-5)


def test_topk_kl_handles_partial_mask():
    torch.manual_seed(4)
    student_logits = torch.randn(2, 5, 30)
    teacher_logits = torch.randn(2, 5, 30)
    idx, lp = topk_from_logits(teacher_logits, top_k=10)
    mask = torch.tensor([[1, 1, 0, 0, 0], [0, 1, 1, 1, 0]], dtype=torch.float)
    loss = topk_forward_kl(student_logits, idx, lp, mask)
    # 5 active positions out of 10.
    assert loss.item() > 0
