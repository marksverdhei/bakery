import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Optional, List

import pytest

from bakery.data import (
    create_dataset,
    prompt_baking_collator,
    load_dataset,
    load_corpus,
    build_system_prompt,
    load_eval_data,
    load_data,
)
from bakery.config import BakeryConfig, DataConfig


def _make_data_config(**kwargs) -> DataConfig:
    return DataConfig(**{k: v for k, v in kwargs.items()})


def _make_baking_config(system_prompt=None) -> BakeryConfig:
    cfg = BakeryConfig.__new__(BakeryConfig)
    cfg.system_prompt = system_prompt
    cfg.system_prompt_file = None
    return cfg


def test_dataset_creation():
    prompts = ["hello", "world"]
    ds = create_dataset(prompts)
    assert len(ds) == 2
    assert ds[0]["user_messages"] == "hello"
    assert "responses" not in ds.column_names


def test_dataset_with_responses():
    prompts = ["hello", "world"]
    responses = ["hi there", "earth"]
    ds = create_dataset(prompts, responses)
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
    data = {
        "pairs": [
            {"prompt": "Hi", "response": "Hello"},
        ]
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name
    try:
        prompts, responses = load_dataset(path)
        assert prompts == ["Hi"]
        assert responses == ["Hello"]
    finally:
        os.unlink(path)


@pytest.mark.parametrize("key", ["data", "samples", "completions", "training_samples",
                                  "training_prompts", "prompts", "questions"])
def test_load_json_wrapped_all_keys(tmp_path, key):
    """All supported wrapper keys unwrap correctly."""
    f = tmp_path / "data.json"
    f.write_text(json.dumps({key: [{"prompt": "Q?", "response": "A."}]}))
    prompts, responses = load_dataset(str(f))
    assert prompts == ["Q?"]
    assert responses == ["A."]


def test_load_json_input_key(tmp_path):
    """'input' key is accepted as prompt key."""
    f = tmp_path / "data.json"
    f.write_text(json.dumps([{"input": "Q?", "response": "A."}]))
    prompts, responses = load_dataset(str(f))
    assert prompts == ["Q?"]
    assert responses == ["A."]


def test_load_json_user_message_key(tmp_path):
    """'user_message' key is accepted as prompt key."""
    f = tmp_path / "data.json"
    f.write_text(json.dumps([{"user_message": "Q?", "completion": "A."}]))
    prompts, responses = load_dataset(str(f))
    assert prompts == ["Q?"]
    assert responses == ["A."]


def test_load_json_output_key(tmp_path):
    """'output' key is accepted as response key."""
    f = tmp_path / "data.json"
    f.write_text(json.dumps([{"prompt": "Q?", "output": "A."}]))
    prompts, responses = load_dataset(str(f))
    assert prompts == ["Q?"]
    assert responses == ["A."]


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------

def test_load_data_from_training_prompts(tmp_path):
    cfg = _make_data_config(training_prompts=["Hello", "World"])
    prompts, responses = load_data(cfg)
    assert prompts == ["Hello", "World"]
    assert responses is None


def test_load_data_from_dataset_file(tmp_path):
    f = tmp_path / "train.json"
    f.write_text(json.dumps(["Q1", "Q2"]))
    cfg = _make_data_config(dataset=str(f))
    prompts, responses = load_data(cfg)
    assert prompts == ["Q1", "Q2"]
    assert responses is None


def test_load_data_prefers_dataset_over_prompts(tmp_path):
    """When both are set, dataset takes priority."""
    f = tmp_path / "train.json"
    f.write_text(json.dumps(["from_file"]))
    cfg = _make_data_config(dataset=str(f), training_prompts=["from_prompts"])
    prompts, _ = load_data(cfg)
    assert prompts == ["from_file"]


def test_load_data_raises_without_source():
    cfg = _make_data_config()
    with pytest.raises(ValueError, match="No training data"):
        load_data(cfg)


# ---------------------------------------------------------------------------
# load_corpus
# ---------------------------------------------------------------------------

def test_load_corpus_none_when_no_file():
    cfg = _make_data_config()
    assert load_corpus(cfg) is None


def test_load_corpus_papers_format(tmp_path):
    papers = [
        {"title": "Paper A", "abstract": "Abstract A."},
        {"title": "Paper B", "abstract": "Abstract B."},
    ]
    f = tmp_path / "corpus.json"
    f.write_text(json.dumps(papers))
    cfg = _make_data_config(corpus_file=str(f), corpus_format="papers")
    result = load_corpus(cfg)
    assert "Paper 1:" in result
    assert "Paper A" in result
    assert "Abstract A" in result
    assert "Paper 2:" in result


def test_load_corpus_text_format_string(tmp_path):
    f = tmp_path / "corpus.json"
    f.write_text(json.dumps("hello world"))
    cfg = _make_data_config(corpus_file=str(f), corpus_format="text")
    result = load_corpus(cfg)
    assert result == "hello world"


def test_load_corpus_text_format_dict(tmp_path):
    f = tmp_path / "corpus.json"
    f.write_text(json.dumps({"text": "the body"}))
    cfg = _make_data_config(corpus_file=str(f), corpus_format="text")
    result = load_corpus(cfg)
    assert result == "the body"


def test_load_corpus_list_format(tmp_path):
    f = tmp_path / "corpus.json"
    f.write_text(json.dumps(["A", "B", "C"]))
    cfg = _make_data_config(corpus_file=str(f), corpus_format="list")
    result = load_corpus(cfg)
    assert "A" in result and "B" in result and "C" in result


def test_load_corpus_custom_format(tmp_path):
    data = {"key": "value"}
    f = tmp_path / "corpus.json"
    f.write_text(json.dumps(data))
    cfg = _make_data_config(corpus_file=str(f), corpus_format="custom")
    result = load_corpus(cfg)
    assert "key" in result


# ---------------------------------------------------------------------------
# build_system_prompt
# ---------------------------------------------------------------------------

def test_build_system_prompt_direct():
    bc = _make_baking_config(system_prompt="You are helpful.")
    dc = _make_data_config()
    result = build_system_prompt(bc, dc)
    assert result == "You are helpful."


def test_build_system_prompt_with_template_and_corpus():
    bc = _make_baking_config()
    bc.system_prompt = None
    dc = _make_data_config(system_prompt_template="Context: {corpus}")
    result = build_system_prompt(bc, dc, corpus="hello world")
    assert result == "Context: hello world"


def test_build_system_prompt_with_corpus_only():
    bc = _make_baking_config()
    bc.system_prompt = None
    dc = _make_data_config()
    result = build_system_prompt(bc, dc, corpus="knowledge base text")
    assert "knowledge base text" in result


def test_build_system_prompt_raises_without_any_source():
    bc = _make_baking_config()
    bc.system_prompt = None
    dc = _make_data_config()
    with pytest.raises(ValueError, match="No system prompt"):
        build_system_prompt(bc, dc)


# ---------------------------------------------------------------------------
# load_eval_data
# ---------------------------------------------------------------------------

def test_load_eval_data_none():
    result = load_eval_data(None)
    assert result == []


def test_load_eval_data_simple(tmp_path):
    data = [{"question": "What is 2+2?", "answer": "4"}]
    f = tmp_path / "eval.json"
    f.write_text(json.dumps(data))
    result = load_eval_data(str(f))
    assert len(result) == 1
    q, kws = result[0]
    assert q == "What is 2+2?"
    assert "4" in kws


def test_load_eval_data_wrapped(tmp_path):
    data = {"evaluation_samples": [{"question": "Q?", "answer": "A"}]}
    f = tmp_path / "eval.json"
    f.write_text(json.dumps(data))
    result = load_eval_data(str(f))
    assert len(result) == 1
    assert result[0][0] == "Q?"


def test_load_eval_data_answer_list(tmp_path):
    """answer can be a list of strings — each should be a keyword."""
    data = [{"question": "Q?", "answer": ["ans1", "ans2"]}]
    f = tmp_path / "eval.json"
    f.write_text(json.dumps(data))
    result = load_eval_data(str(f))
    _, kws = result[0]
    assert "ans1" in kws
    assert "ans2" in kws


def test_load_eval_data_input_key(tmp_path):
    """'input' key is an alias for 'question'."""
    data = [{"input": "Hi?", "answer": "Hello"}]
    f = tmp_path / "eval.json"
    f.write_text(json.dumps(data))
    result = load_eval_data(str(f))
    assert result[0][0] == "Hi?"


def test_load_eval_data_keywords_lowercased(tmp_path):
    data = [{"question": "Q?", "answer": "Paris"}]
    f = tmp_path / "eval.json"
    f.write_text(json.dumps(data))
    _, kws = load_eval_data(str(f))[0]
    assert "paris" in kws


@pytest.mark.parametrize("key", ["test_samples", "eval", "qa_pairs"])
def test_load_eval_data_wrapped_keys(tmp_path, key):
    """All supported wrapper keys for eval data."""
    data = {key: [{"question": "Q?", "answer": "A"}]}
    f = tmp_path / "eval.json"
    f.write_text(json.dumps(data))
    result = load_eval_data(str(f))
    assert len(result) == 1
    assert result[0][0] == "Q?"
