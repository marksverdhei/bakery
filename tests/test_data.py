import json
import os
import tempfile

from bakery.data import PromptBakingDataset, prompt_baking_collator, load_dataset


def test_dataset_creation():
    prompts = ["hello", "world"]
    ds = PromptBakingDataset(prompts)
    assert len(ds) == 2
    assert ds[0]["user_messages"] == "hello"
    assert "responses" not in ds.column_names


def test_dataset_with_responses():
    prompts = ["hello", "world"]
    responses = ["hi there", "earth"]
    ds = PromptBakingDataset(prompts, responses)
    assert len(ds) == 2
    assert ds[0]["responses"] == "hi there"


def test_collator():
    features = [
        {"user_messages": "hello", "responses": "hi"},
        {"user_messages": "world", "responses": "earth"},
    ]
    batch = prompt_baking_collator(features)
    assert batch["user_messages"] == ["hello", "world"]
    assert batch["responses"] == ["hi", "earth"]


def test_collator_handles_lists():
    features = [
        {"user_messages": ["a", "b"], "responses": ["c", "d"]},
    ]
    batch = prompt_baking_collator(features)
    assert batch["user_messages"] == ["a", "b"]
    assert batch["responses"] == ["c", "d"]


def test_load_json_prompts_only():
    """List of strings → prompts only, no responses."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(["What is AI?", "Explain gravity."], f)
        path = f.name
    try:
        prompts, responses = load_dataset(path)
        assert prompts == ["What is AI?", "Explain gravity."]
        assert responses is None
    finally:
        os.unlink(path)


def test_load_json_paired():
    """List of dicts with prompt+response → paired data."""
    data = [
        {"prompt": "What is AI?", "response": "AI is..."},
        {"prompt": "Explain gravity.", "response": "Gravity is..."},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name
    try:
        prompts, responses = load_dataset(path)
        assert prompts == ["What is AI?", "Explain gravity."]
        assert responses == ["AI is...", "Gravity is..."]
    finally:
        os.unlink(path)


def test_load_json_prompts_only_dicts():
    """List of dicts with only prompt keys (no response) → prompts only."""
    data = [
        {"question": "What is AI?"},
        {"question": "Explain gravity."},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name
    try:
        prompts, responses = load_dataset(path)
        assert prompts == ["What is AI?", "Explain gravity."]
        assert responses is None
    finally:
        os.unlink(path)


def test_load_json_wrapped():
    """Nested format like {"pairs": [...]} → unwrapped correctly."""
    data = {"pairs": [
        {"prompt": "Hi", "response": "Hello"},
    ]}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name
    try:
        prompts, responses = load_dataset(path)
        assert prompts == ["Hi"]
        assert responses == ["Hello"]
    finally:
        os.unlink(path)
