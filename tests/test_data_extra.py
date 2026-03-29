"""Additional tests for bakery.data — nested keys, alt field names, collator edge cases."""

import json
import os
import tempfile

import pytest

from bakery.data import load_dataset, prompt_baking_collator, load_data
from bakery.config import DataConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(data):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, f)
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# _load_json — nested container keys
# ---------------------------------------------------------------------------

class TestLoadJsonNestedKeys:
    def _roundtrip(self, wrapper_key, items):
        path = _write_json({wrapper_key: items})
        try:
            return load_dataset(path)
        finally:
            os.unlink(path)

    def test_data_key_unwrapped(self):
        prompts, responses = self._roundtrip(
            "data", [{"prompt": "Q", "response": "A"}]
        )
        assert prompts == ["Q"]
        assert responses == ["A"]

    def test_samples_key_unwrapped(self):
        prompts, _ = self._roundtrip("samples", [{"prompt": "Q"}])
        assert prompts == ["Q"]

    def test_completions_key_unwrapped(self):
        prompts, responses = self._roundtrip(
            "completions", [{"prompt": "Q", "completion": "A"}]
        )
        assert prompts == ["Q"]
        assert responses == ["A"]

    def test_training_samples_key_unwrapped(self):
        prompts, _ = self._roundtrip("training_samples", [{"question": "Q?"}])
        assert prompts == ["Q?"]

    def test_training_prompts_key_string_list(self):
        prompts, responses = self._roundtrip("training_prompts", ["p1", "p2"])
        assert prompts == ["p1", "p2"]
        assert responses is None

    def test_prompts_key_string_list(self):
        prompts, responses = self._roundtrip("prompts", ["a", "b"])
        assert prompts == ["a", "b"]
        assert responses is None

    def test_questions_key_unwrapped(self):
        prompts, _ = self._roundtrip("questions", [{"question": "What?"}])
        assert prompts == ["What?"]


# ---------------------------------------------------------------------------
# _load_json — alternative prompt field names
# ---------------------------------------------------------------------------

class TestLoadJsonAltPromptFields:
    def _load(self, items):
        path = _write_json(items)
        try:
            return load_dataset(path)
        finally:
            os.unlink(path)

    def test_user_message_field(self):
        prompts, _ = self._load([{"user_message": "Hello"}])
        assert prompts == ["Hello"]

    def test_input_field(self):
        prompts, _ = self._load([{"input": "Translate this"}])
        assert prompts == ["Translate this"]

    def test_question_field(self):
        prompts, _ = self._load([{"question": "Why?"}])
        assert prompts == ["Why?"]

    def test_completion_response_field(self):
        _, responses = self._load([{"prompt": "Q", "completion": "C"}])
        assert responses == ["C"]

    def test_output_response_field(self):
        _, responses = self._load([{"prompt": "Q", "output": "O"}])
        assert responses == ["O"]

    def test_items_with_no_prompt_are_skipped(self):
        """Items with no recognized prompt key should be skipped."""
        items = [
            {"unknown_field": "x"},
            {"prompt": "Valid", "response": "R"},
        ]
        prompts, responses = self._load(items)
        # first item has no prompt → skipped; only second kept
        assert prompts == ["Valid"]
        assert responses == ["R"]


# ---------------------------------------------------------------------------
# prompt_baking_collator — edge cases
# ---------------------------------------------------------------------------

class TestCollatorEdgeCases:
    def test_missing_responses_key(self):
        """Features without 'responses' should produce empty responses list."""
        features = [{"user_messages": "hello"}]
        batch = prompt_baking_collator(features)
        assert batch["user_messages"] == ["hello"]
        assert batch["responses"] == []

    def test_none_user_messages_skipped(self):
        """None user_messages should not be added."""
        features = [{"user_messages": None, "responses": None}]
        batch = prompt_baking_collator(features)
        assert batch["user_messages"] == []
        assert batch["responses"] == []

    def test_mixed_str_and_list(self):
        """Mix of str and list features is handled."""
        features = [
            {"user_messages": "a", "responses": "x"},
            {"user_messages": ["b", "c"], "responses": ["y", "z"]},
        ]
        batch = prompt_baking_collator(features)
        assert batch["user_messages"] == ["a", "b", "c"]
        assert batch["responses"] == ["x", "y", "z"]

    def test_empty_features_list(self):
        batch = prompt_baking_collator([])
        assert batch["user_messages"] == []
        assert batch["responses"] == []


# ---------------------------------------------------------------------------
# load_data — routing logic
# ---------------------------------------------------------------------------

class TestLoadData:
    def test_uses_dataset_when_set(self, tmp_path):
        path = str(tmp_path / "data.json")
        with open(path, "w") as f:
            json.dump(["p1", "p2"], f)
        config = DataConfig(dataset=path)
        prompts, responses = load_data(config)
        assert prompts == ["p1", "p2"]
        assert responses is None

    def test_uses_training_prompts_when_no_dataset(self):
        config = DataConfig(training_prompts=["hello", "world"])
        prompts, responses = load_data(config)
        assert prompts == ["hello", "world"]
        assert responses is None

    def test_raises_when_no_data_configured(self):
        config = DataConfig()
        with pytest.raises(ValueError, match="No training data configured"):
            load_data(config)

    def test_dataset_takes_priority_over_training_prompts(self, tmp_path):
        path = str(tmp_path / "data.json")
        with open(path, "w") as f:
            json.dump(["from_file"], f)
        config = DataConfig(dataset=path, training_prompts=["ignored"])
        prompts, _ = load_data(config)
        assert prompts == ["from_file"]
