"""Tests for OpenAIAPITeacher.score against a mocked vLLM-style server.

We swap httpx.Client out for a stub that returns hand-crafted logprobs
payloads, so this runs offline / in CI.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import AutoTokenizer

from bakery.teachers.openai_compat import OpenAIAPITeacher


def _mock_response(payload):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=payload)
    return resp


def _mock_client(payload):
    client = MagicMock()
    client.post = MagicMock(return_value=_mock_response(payload))
    client.close = MagicMock()
    return client


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return tok


# ---------- guardrails ----------


def test_score_requires_student_tokenizer():
    teacher = OpenAIAPITeacher(api_base="http://x", model="m")
    with pytest.raises(RuntimeError, match="student tokenizer"):
        teacher.score(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.ones(1, 3),
            top_k=4,
        )


def test_score_requires_echo_support():
    teacher = OpenAIAPITeacher(api_base="http://x", model="m", echo_supported=False)
    with pytest.raises(NotImplementedError, match="echo"):
        teacher.score(
            input_ids=torch.tensor([[1]]),
            attention_mask=torch.ones(1, 1),
            top_k=4,
        )


# ---------- score happy path ----------


def _payload_for(tokens_and_topks):
    """Helper: build a vLLM-style payload from [(token_string, {tok: lp})]."""
    return {
        "choices": [
            {
                "logprobs": {
                    "tokens": [t for t, _ in tokens_and_topks],
                    "top_logprobs": [tp for _, tp in tokens_and_topks],
                }
            }
        ]
    }


def test_score_returns_topklogprobs_shape(gpt2_tokenizer):
    teacher = OpenAIAPITeacher(api_base="http://x", model="m")
    teacher.set_student_tokenizer(gpt2_tokenizer)

    # Three echoed positions; position 0 has no top_logprobs (typical), 1+2 do.
    payload = _payload_for(
        [
            ("Hello", None),
            (",", {",": -0.1, " world": -2.0, "!": -3.5}),
            (" world", {" world": -0.2, "!": -1.5, ".": -3.0}),
        ]
    )
    with patch.object(teacher, "_client", return_value=_mock_client(payload)):
        out = teacher.score(
            input_ids=torch.tensor([[15496, 11, 995]]),
            attention_mask=torch.ones(1, 3),
            top_k=3,
        )
    assert out.indices.shape == (1, 3, 3)
    assert out.values.shape == (1, 3, 3)


def test_score_values_renormalized(gpt2_tokenizer):
    """After re-encoding + renorm, exp(values) at scored positions sum ≈ 1."""
    teacher = OpenAIAPITeacher(api_base="http://x", model="m")
    teacher.set_student_tokenizer(gpt2_tokenizer)
    payload = _payload_for(
        [
            (",", None),
            (
                " world",
                {" world": -0.2, "!": -1.5, ".": -3.0, "Hello": -4.0},
            ),
        ]
    )
    with patch.object(teacher, "_client", return_value=_mock_client(payload)):
        out = teacher.score(
            input_ids=torch.tensor([[11, 995]]),
            attention_mask=torch.ones(1, 2),
            top_k=4,
        )
    # Position 1 should have real renormalized logprobs.
    s = out.values[0, 1].exp().sum().item()
    assert 0.99 <= s <= 1.01


def test_score_skips_null_first_position(gpt2_tokenizer):
    """Echo's first position usually has null top_logprobs — we skip it."""
    teacher = OpenAIAPITeacher(api_base="http://x", model="m")
    teacher.set_student_tokenizer(gpt2_tokenizer)
    payload = _payload_for(
        [
            ("Hello", None),
            (",", {",": -0.1, "!": -2.0}),
        ]
    )
    with patch.object(teacher, "_client", return_value=_mock_client(payload)):
        out = teacher.score(
            input_ids=torch.tensor([[15496, 11]]),
            attention_mask=torch.ones(1, 2),
            top_k=2,
        )
    # Position 0 stays at default placeholder values (-1e9).
    assert (out.values[0, 0] <= -1e8).all()
    # Position 1 has real values.
    assert out.values[0, 1].max().item() > -1e3


def test_score_strips_left_padding(gpt2_tokenizer):
    """Padded prefix tokens should not be sent to the server."""
    teacher = OpenAIAPITeacher(api_base="http://x", model="m")
    teacher.set_student_tokenizer(gpt2_tokenizer)
    payload = _payload_for([(",", None), (" world", {" world": -0.2, "!": -1.0})])
    captured = {}

    def fake_post(*args, **kwargs):
        captured["json"] = kwargs.get("json")
        return _mock_response(payload)

    client = MagicMock()
    client.post = fake_post
    client.close = MagicMock()
    with patch.object(teacher, "_client", return_value=client):
        out = teacher.score(
            input_ids=torch.tensor([[0, 0, 11, 995]]),
            attention_mask=torch.tensor([[0, 0, 1, 1]]),
            top_k=2,
        )
    # Only 2 real tokens should have been sent (left padding stripped).
    assert captured["json"]["prompt"] == [11, 995]
    # The output preserves the original shape, padding positions left untouched.
    assert out.indices.shape == (1, 4, 2)


def test_score_empty_top_logprobs_position(gpt2_tokenizer):
    """A position whose top_logprobs is {} should not blow up."""
    teacher = OpenAIAPITeacher(api_base="http://x", model="m")
    teacher.set_student_tokenizer(gpt2_tokenizer)
    payload = _payload_for(
        [
            ("Hello", None),
            (",", {}),
            (" world", {" world": -0.2, "!": -1.0}),
        ]
    )
    with patch.object(teacher, "_client", return_value=_mock_client(payload)):
        out = teacher.score(
            input_ids=torch.tensor([[15496, 11, 995]]),
            attention_mask=torch.ones(1, 3),
            top_k=3,
        )
    # No crash, and the empty position keeps placeholder values.
    assert (out.values[0, 1] <= -1e8).all()
    assert out.values[0, 2].max().item() > -1e3


# ---------- generate ----------


def test_generate_sends_messages():
    teacher = OpenAIAPITeacher(api_base="http://x", model="m")
    captured = {}

    def fake_post(*args, **kwargs):
        captured["json"] = kwargs.get("json")
        return _mock_response({"choices": [{"message": {"content": "the answer"}}]})

    client = MagicMock()
    client.post = fake_post
    client.close = MagicMock()
    with patch.object(teacher, "_client", return_value=client):
        out = teacher.generate(
            [{"role": "user", "content": "What is the answer?"}],
            max_new_tokens=8,
        )
    assert out == "the answer"
    assert captured["json"]["messages"][0]["content"] == "What is the answer?"
    assert captured["json"]["max_tokens"] == 8
