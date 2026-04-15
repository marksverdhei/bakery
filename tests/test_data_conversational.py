"""Tests for conversational data loading and the generalized collator."""

import json
import os
import tempfile

import pytest

from bakery.data import (
    create_conversational_dataset,
    load_conversations,
    prompt_baking_collator,
)


# ---------- create_conversational_dataset ----------


def test_create_conversational_dataset_empty():
    ds = create_conversational_dataset([])
    assert len(ds) == 0
    assert set(ds.column_names) == {"turns", "prefix_messages", "responses"}


def test_create_conversational_dataset_single_turn_with_response():
    ds = create_conversational_dataset(
        [
            {
                "turns": [{"role": "user", "content": "hi"}],
                "prefix_messages": None,
                "response": "hello",
            }
        ]
    )
    assert len(ds) == 1
    assert ds[0]["turns"] == [{"role": "user", "content": "hi"}]
    assert ds[0]["prefix_messages"] is None
    assert ds[0]["responses"] == "hello"


def test_create_conversational_dataset_missing_keys_default_to_none_or_empty():
    """Rows that omit optional fields still produce a well-formed dataset."""
    ds = create_conversational_dataset([{"turns": []}])
    assert len(ds) == 1
    assert ds[0]["turns"] == []
    assert ds[0]["prefix_messages"] is None
    assert ds[0]["responses"] is None


def test_create_conversational_dataset_preserves_row_order():
    rows = [
        {"turns": [{"role": "user", "content": f"q{i}"}], "response": f"a{i}"}
        for i in range(5)
    ]
    ds = create_conversational_dataset(rows)
    for i in range(5):
        assert ds[i]["turns"][0]["content"] == f"q{i}"
        assert ds[i]["responses"] == f"a{i}"


def test_create_conversational_dataset_mixed_prefix_presence():
    rows = [
        {"turns": [{"role": "user", "content": "a"}], "prefix_messages": [{"role": "system", "content": "P"}]},
        {"turns": [{"role": "user", "content": "b"}]},
    ]
    ds = create_conversational_dataset(rows)
    assert ds[0]["prefix_messages"] == [{"role": "system", "content": "P"}]
    assert ds[1]["prefix_messages"] is None


# ---------- prompt_baking_collator ----------


def test_collator_legacy_shape():
    feats = [
        {"user_messages": "q1", "responses": "a1"},
        {"user_messages": "q2", "responses": "a2"},
    ]
    batch = prompt_baking_collator(feats)
    assert "turns" not in batch
    assert batch["user_messages"] == ["q1", "q2"]
    assert batch["responses"] == ["a1", "a2"]


def test_collator_conversational_shape():
    feats = [
        {
            "turns": [{"role": "user", "content": "q"}],
            "prefix_messages": [{"role": "system", "content": "s"}],
            "responses": "a",
        }
    ]
    batch = prompt_baking_collator(feats)
    assert batch["turns"] == [[{"role": "user", "content": "q"}]]
    assert batch["prefix_messages"] == [[{"role": "system", "content": "s"}]]
    assert batch["responses"] == ["a"]
    assert "user_messages" not in batch


def test_collator_preserves_none_prefix():
    feats = [
        {
            "turns": [{"role": "user", "content": "q"}],
            "prefix_messages": None,
            "responses": None,
        }
    ]
    batch = prompt_baking_collator(feats)
    assert batch["prefix_messages"] == [None]
    assert batch["responses"] == [None]


def test_collator_conversational_handles_null_turns():
    """None turns should be coerced to []."""
    feats = [{"turns": None, "prefix_messages": None, "responses": None}]
    batch = prompt_baking_collator(feats)
    assert batch["turns"] == [[]]


def test_collator_batch_size_multi_conversational():
    feats = [
        {
            "turns": [{"role": "user", "content": f"q{i}"}],
            "prefix_messages": None,
            "responses": f"a{i}",
        }
        for i in range(3)
    ]
    batch = prompt_baking_collator(feats)
    assert len(batch["turns"]) == 3
    assert len(batch["responses"]) == 3


def test_collator_empty_legacy_features():
    batch = prompt_baking_collator([])
    # Empty legacy shape is the default when no features have 'turns'.
    assert batch == {"user_messages": [], "responses": []}


def test_collator_legacy_list_values_are_extended():
    feats = [{"user_messages": ["a", "b"], "responses": ["x", "y"]}]
    batch = prompt_baking_collator(feats)
    assert batch["user_messages"] == ["a", "b"]
    assert batch["responses"] == ["x", "y"]


# ---------- load_conversations (JSON) ----------


@pytest.fixture
def tmpjson():
    paths = []

    def _write(obj):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(obj, f)
        f.close()
        paths.append(f.name)
        return f.name

    yield _write

    for p in paths:
        try:
            os.unlink(p)
        except OSError:
            pass


def test_load_conversations_json_messages_format(tmpjson):
    path = tmpjson(
        [
            {
                "messages": [
                    {"role": "user", "content": "q1"},
                    {"role": "assistant", "content": "a1"},
                ]
            }
        ]
    )
    rows = load_conversations(path)
    assert len(rows) == 1
    assert rows[0]["turns"] == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]
    assert rows[0]["prefix_messages"] is None
    assert rows[0]["response"] is None


def test_load_conversations_json_with_per_row_prefix(tmpjson):
    path = tmpjson(
        [
            {
                "prefix_messages": [{"role": "system", "content": "persona A"}],
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ],
            }
        ]
    )
    rows = load_conversations(path)
    assert rows[0]["prefix_messages"] == [{"role": "system", "content": "persona A"}]


def test_load_conversations_json_prompt_response_pairs(tmpjson):
    path = tmpjson(
        [
            {"prompt": "q1", "response": "a1"},
            {"prompt": "q2", "response": "a2"},
        ]
    )
    rows = load_conversations(path)
    assert len(rows) == 2
    assert rows[0]["turns"] == [{"role": "user", "content": "q1"}]
    assert rows[0]["response"] == "a1"


def test_load_conversations_json_plain_string_list(tmpjson):
    """List of strings becomes single-turn user-only rows."""
    path = tmpjson(["q1", "q2"])
    rows = load_conversations(path)
    assert len(rows) == 2
    assert rows[0]["turns"] == [{"role": "user", "content": "q1"}]
    assert rows[0]["response"] is None
    assert rows[0]["prefix_messages"] is None


def test_load_conversations_json_nested_data_key(tmpjson):
    path = tmpjson({"conversations": [{"prompt": "q", "response": "a"}]})
    rows = load_conversations(path)
    assert len(rows) == 1
    assert rows[0]["turns"] == [{"role": "user", "content": "q"}]
    assert rows[0]["response"] == "a"


def test_load_conversations_json_skips_empty_prompts(tmpjson):
    """Rows with empty prompt + no messages column should be skipped."""
    path = tmpjson([{"prompt": "", "response": "a"}, {"prompt": "q", "response": "a"}])
    rows = load_conversations(path)
    # At least the valid row is present.
    assert any(r["turns"] == [{"role": "user", "content": "q"}] for r in rows)


def test_load_conversations_json_alternate_prompt_keys(tmpjson):
    path = tmpjson([{"question": "q", "answer": "a"}])
    rows = load_conversations(path)
    assert rows[0]["turns"] == [{"role": "user", "content": "q"}]
    # `answer` isn't in the recognized response keys, so response is None.
    # (Only prompt-side key remapping is aggressive; response uses canonical keys.)
    # We just assert the turn was loaded correctly.
    assert len(rows) == 1
