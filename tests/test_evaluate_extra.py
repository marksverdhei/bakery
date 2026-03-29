"""Additional tests for bakery.evaluate — system prompt, multi-pair, result structure."""

from unittest.mock import MagicMock

import torch
import pytest

from bakery.evaluate import evaluate_model


def _make_tokenizer():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = (
        "{% for m in messages %}"
        "{{ m['role'] }}: {{ m['content'] }}\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}"
    )
    return tokenizer


def _make_model(response_text: str, tokenizer):
    model = MagicMock()
    model.training = False
    model.device = torch.device("cpu")

    prompt_ids = tokenizer("x", return_tensors="pt")["input_ids"]
    response_ids = tokenizer(response_text, add_special_tokens=False, return_tensors="pt")["input_ids"]

    def fake_generate(**kwargs):
        return torch.cat([kwargs.get("input_ids", prompt_ids), response_ids], dim=1)

    model.generate = MagicMock(side_effect=fake_generate)
    return model


class TestEvaluateModelResults:
    def test_empty_pairs_results_is_empty_list(self):
        result = evaluate_model(None, None, [])
        assert result["results"] == []

    def test_result_entry_has_required_keys(self):
        tokenizer = _make_tokenizer()
        model = _make_model("answer", tokenizer)
        result = evaluate_model(model, tokenizer, [("q", ["answer"])])
        entry = result["results"][0]
        assert "question" in entry
        assert "response" in entry
        assert "correct" in entry

    def test_result_entry_question_matches_input(self):
        tokenizer = _make_tokenizer()
        model = _make_model("answer", tokenizer)
        result = evaluate_model(model, tokenizer, [("What is 2+2?", ["answer"])])
        assert result["results"][0]["question"] == "What is 2+2?"

    def test_result_entry_response_is_lowercased(self):
        tokenizer = _make_tokenizer()
        model = _make_model("UPPERCASE ANSWER", tokenizer)
        result = evaluate_model(model, tokenizer, [("q", ["uppercase"])])
        # response is lowercased internally
        assert result["results"][0]["response"] == result["results"][0]["response"].lower()

    def test_multiple_pairs_correct_count(self):
        """2 correct out of 3 → accuracy = 2/3."""
        tokenizer = _make_tokenizer()
        model = _make_model("paris", tokenizer)
        qa_pairs = [
            ("Capital of France?", ["paris"]),
            ("Capital of Germany?", ["paris"]),   # wrong keyword, but response contains "paris"
            ("Capital of UK?", ["london"]),        # "paris" ≠ "london" → wrong
        ]
        result = evaluate_model(model, tokenizer, qa_pairs)
        assert result["total"] == 3
        assert result["correct"] == 2
        assert result["accuracy"] == pytest.approx(2 / 3)

    def test_result_list_length_matches_pairs(self):
        tokenizer = _make_tokenizer()
        model = _make_model("answer", tokenizer)
        qa_pairs = [("q1", ["a"]), ("q2", ["b"]), ("q3", ["c"])]
        result = evaluate_model(model, tokenizer, qa_pairs)
        assert len(result["results"]) == 3


class TestEvaluateSystemPrompt:
    def test_system_prompt_passed_to_model(self):
        """When system_prompt is set, tokenizer should receive system message."""
        tokenizer = _make_tokenizer()
        model = _make_model("response", tokenizer)

        # Patch tokenizer to capture what gets passed
        original_apply = tokenizer.apply_chat_template
        captured_messages = []

        def capturing_apply(messages, **kwargs):
            captured_messages.extend(messages)
            return original_apply(messages, **kwargs)

        tokenizer.apply_chat_template = capturing_apply

        evaluate_model(model, tokenizer, [("question", ["response"])], system_prompt="Be helpful.")

        roles = [m["role"] for m in captured_messages]
        assert "system" in roles
        system_content = next(m["content"] for m in captured_messages if m["role"] == "system")
        assert system_content == "Be helpful."

    def test_no_system_message_when_prompt_not_set(self):
        tokenizer = _make_tokenizer()
        model = _make_model("response", tokenizer)

        original_apply = tokenizer.apply_chat_template
        captured_messages = []

        def capturing_apply(messages, **kwargs):
            captured_messages.extend(messages)
            return original_apply(messages, **kwargs)

        tokenizer.apply_chat_template = capturing_apply

        evaluate_model(model, tokenizer, [("question", ["response"])])

        roles = [m["role"] for m in captured_messages]
        assert "system" not in roles


class TestEvaluateKeywordMatching:
    def test_any_keyword_sufficient_for_correct(self):
        """If any keyword matches, the answer is correct."""
        tokenizer = _make_tokenizer()
        model = _make_model("the answer is berlin", tokenizer)
        # First keyword doesn't match, second does
        qa_pairs = [("Capital?", ["paris", "berlin"])]
        result = evaluate_model(model, tokenizer, qa_pairs)
        assert result["results"][0]["correct"] is True

    def test_keyword_matching_is_case_insensitive(self):
        """Keywords from load_eval_data are already lowercased; response is lowercased."""
        tokenizer = _make_tokenizer()
        model = _make_model("PARIS is the answer", tokenizer)
        qa_pairs = [("Capital?", ["paris"])]
        result = evaluate_model(model, tokenizer, qa_pairs)
        assert result["results"][0]["correct"] is True
