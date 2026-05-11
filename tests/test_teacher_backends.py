"""Tests for TeacherBackend implementations and the make_teacher factory."""

import pytest
import torch

from bakery.config import TeacherConfig
from bakery.teachers import HFTeacher, TopKLogprobs, make_teacher


# ---------- TopKLogprobs / topk_from_logits ----------


def test_topklogprobs_is_dense_heuristic():
    """is_dense triggers only when K is large (gpt2 vocab-ish)."""
    small = TopKLogprobs(
        indices=torch.zeros(1, 1, 8, dtype=torch.long),
        values=torch.zeros(1, 1, 8),
        attention_mask=torch.ones(1, 1),
    )
    assert not small.is_dense

    big = TopKLogprobs(
        indices=torch.zeros(1, 1, 50000, dtype=torch.long),
        values=torch.zeros(1, 1, 50000),
        attention_mask=torch.ones(1, 1),
    )
    assert big.is_dense


# ---------- make_teacher dispatch ----------


def test_make_teacher_local_toggle_returns_none():
    """local-toggle (the default) means: no external teacher."""
    cfg = TeacherConfig()
    assert make_teacher(cfg) is None


def test_make_teacher_none_aliases():
    """Several aliases all resolve to local-toggle."""
    for alias in ("local-toggle", "local", "self", "none", ""):
        cfg = TeacherConfig(teacher_backend=alias)
        assert make_teacher(cfg) is None


def test_make_teacher_unknown_raises():
    cfg = TeacherConfig(teacher_backend="not-a-real-backend")
    with pytest.raises(ValueError, match="Unknown teacher backend"):
        make_teacher(cfg)


def test_make_teacher_hf_requires_model_name():
    cfg = TeacherConfig(teacher_backend="hf")
    with pytest.raises(ValueError, match="model_name_or_path"):
        make_teacher(cfg)


def test_make_teacher_openai_requires_model_and_base():
    cfg = TeacherConfig(teacher_backend="openai")
    with pytest.raises((ValueError, ImportError)):
        make_teacher(cfg)


# ---------- HFTeacher (uses tiny GPT-2) ----------


@pytest.fixture(scope="module")
def hf_teacher():
    """A real HFTeacher backed by gpt2 — tiny enough to run on CPU."""
    return HFTeacher(model_name_or_path="gpt2", torch_dtype="float32")


def test_hf_teacher_score_returns_topklogprobs(hf_teacher):
    """score() returns top-k indices, renormalized logprobs, attention mask."""
    input_ids = torch.tensor([[15496, 11, 995]])  # "Hello, world"
    attn = torch.ones_like(input_ids)
    out = hf_teacher.score(input_ids, attn, top_k=10)
    assert isinstance(out, TopKLogprobs)
    assert out.indices.shape == (1, 3, 10)
    assert out.values.shape == (1, 3, 10)
    # Renormalized over top-k → exp().sum ≈ 1.
    sums = out.values.exp().sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_hf_teacher_score_respects_top_k(hf_teacher):
    """top_k smaller than vocab → that's what we get."""
    input_ids = torch.tensor([[15496, 11]])
    attn = torch.ones_like(input_ids)
    out = hf_teacher.score(input_ids, attn, top_k=5)
    assert out.indices.shape[-1] == 5


def test_hf_teacher_score_full_vocab(hf_teacher):
    """Asking for K > vocab caps at vocab size."""
    input_ids = torch.tensor([[15496]])
    attn = torch.ones_like(input_ids)
    out = hf_teacher.score(input_ids, attn, top_k=999_999)
    # GPT-2 vocab is 50257.
    assert out.indices.shape[-1] == 50257


def test_hf_teacher_generate_returns_string(hf_teacher):
    """generate() produces a string, even from a trivial chat message."""
    # gpt2 has no chat template by default; set a minimal one for this test.
    hf_teacher.tokenizer.chat_template = (
        "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
    )
    out = hf_teacher.generate([{"role": "user", "content": "Hello"}], max_new_tokens=3)
    assert isinstance(out, str)


def test_hf_teacher_score_no_grad(hf_teacher):
    """Teacher params are frozen; scoring should not produce gradients."""
    input_ids = torch.tensor([[15496, 11]])
    attn = torch.ones_like(input_ids)
    out = hf_teacher.score(input_ids, attn, top_k=4)
    # Values are detached (no_grad path).
    assert not out.values.requires_grad


def test_hf_teacher_score_padding_mask_propagated(hf_teacher):
    """attention_mask is returned on the TopKLogprobs."""
    input_ids = torch.tensor([[15496, 11, 0, 0]])
    attn = torch.tensor([[1, 1, 0, 0]])
    out = hf_teacher.score(input_ids, attn, top_k=4)
    assert out.attention_mask.tolist() == [[1, 1, 0, 0]]


# ---------- HFTeacher init guardrails ----------


def test_hf_teacher_requires_model_name():
    with pytest.raises(ValueError, match="model_name_or_path"):
        HFTeacher(model_name_or_path="")
