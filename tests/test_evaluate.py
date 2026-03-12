import re
from unittest.mock import MagicMock

import torch

from bakery.evaluate import evaluate_model


def test_think_tag_stripping():
    """Verify that <think>...</think> blocks are fully removed, not just tags."""
    response = "<think>internal reasoning here</think>The actual answer"
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    assert cleaned == "The actual answer"


def test_think_tag_multiline():
    response = "<think>\nstep 1\nstep 2\n</think>\nfinal answer"
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    assert cleaned == "final answer"


def test_evaluate_empty_pairs():
    result = evaluate_model(None, None, [], "test")
    assert result["accuracy"] == 0
    assert result["total"] == 0


def _make_mock_model(response_text: str, tokenizer):
    """Create a mock model that 'generates' a fixed response."""
    model = MagicMock()
    model.training = False
    model.device = torch.device("cpu")

    prompt_ids = tokenizer("prompt placeholder", return_tensors="pt")["input_ids"]
    response_ids = tokenizer(
        response_text, add_special_tokens=False, return_tensors="pt"
    )["input_ids"]

    def fake_generate(**kwargs):
        return torch.cat([kwargs.get("input_ids", prompt_ids), response_ids], dim=1)

    model.generate = MagicMock(side_effect=fake_generate)
    return model


def test_evaluate_model_correct_keyword():
    """evaluate_model returns correct=True when response contains expected keyword."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = (
        "{% for m in messages %}"
        "{{ m['role'] }}: {{ m['content'] }}\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}"
    )

    model = _make_mock_model("The capital of France is Paris.", tokenizer)
    qa_pairs = [("What is the capital of France?", ["paris"])]

    result = evaluate_model(model, tokenizer, qa_pairs, "test")
    assert result["total"] == 1
    assert result["correct"] == 1
    assert result["accuracy"] == 1.0
    assert result["results"][0]["correct"] is True


def test_evaluate_model_incorrect_keyword():
    """evaluate_model returns correct=False when keyword is missing."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = (
        "{% for m in messages %}"
        "{{ m['role'] }}: {{ m['content'] }}\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}"
    )

    model = _make_mock_model("I don't know the answer.", tokenizer)
    qa_pairs = [("What is the capital of France?", ["paris"])]

    result = evaluate_model(model, tokenizer, qa_pairs, "test")
    assert result["correct"] == 0
    assert result["accuracy"] == 0.0


def test_evaluate_model_with_think_tags():
    """evaluate_model strips <think> blocks before keyword matching."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = (
        "{% for m in messages %}"
        "{{ m['role'] }}: {{ m['content'] }}\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}"
    )

    model = _make_mock_model(
        "<think>Let me think... Paris is wrong</think>The answer is Berlin.",
        tokenizer,
    )
    # "paris" appears only inside <think> tags, so it should NOT match
    qa_pairs = [("What is the capital?", ["paris"])]

    result = evaluate_model(model, tokenizer, qa_pairs, "test")
    assert result["results"][0]["correct"] is False


def test_evaluate_model_restores_training_mode():
    """evaluate_model restores model.train() if it was training before."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = (
        "{% for m in messages %}"
        "{{ m['role'] }}: {{ m['content'] }}\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}"
    )

    model = _make_mock_model("answer", tokenizer)
    model.training = True

    evaluate_model(model, tokenizer, [("q", ["answer"])], "test")
    model.train.assert_called()
