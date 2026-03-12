import json
import os
import tempfile

from bakery.config import DataConfig
from bakery.data import load_corpus


def test_load_corpus_none():
    data = DataConfig()
    assert load_corpus(data) is None


def test_load_corpus_papers():
    papers = [
        {"title": "Paper A", "abstract": "Abstract A"},
        {"title": "Paper B", "abstract": "Abstract B"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(papers, f)
        path = f.name
    try:
        data = DataConfig(corpus_file=path, corpus_format="papers")
        result = load_corpus(data)
        assert "Paper A" in result
        assert "Abstract B" in result
    finally:
        os.unlink(path)


def test_load_corpus_list():
    items = ["fact one", "fact two"]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(items, f)
        path = f.name
    try:
        data = DataConfig(corpus_file=path, corpus_format="list")
        result = load_corpus(data)
        assert "fact one" in result
        assert "fact two" in result
    finally:
        os.unlink(path)


def test_load_corpus_text_string():
    """text format with a bare JSON string."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump("This is the full corpus text.", f)
        path = f.name
    try:
        data = DataConfig(corpus_file=path, corpus_format="text")
        result = load_corpus(data)
        assert result == "This is the full corpus text."
    finally:
        os.unlink(path)


def test_load_corpus_text_dict():
    """text format with {"text": "..."}."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"text": "corpus from dict"}, f)
        path = f.name
    try:
        data = DataConfig(corpus_file=path, corpus_format="text")
        result = load_corpus(data)
        assert result == "corpus from dict"
    finally:
        os.unlink(path)


def test_load_corpus_text_dict_missing_key():
    """text format with a dict that has no 'text' key returns empty string."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"other_key": "value"}, f)
        path = f.name
    try:
        data = DataConfig(corpus_file=path, corpus_format="text")
        result = load_corpus(data)
        assert result == ""
    finally:
        os.unlink(path)


def test_load_corpus_custom_fallback():
    """Unknown format falls back to json.dumps of the raw data."""
    payload = {"custom_field": "custom_value", "nested": [1, 2, 3]}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        path = f.name
    try:
        data = DataConfig(corpus_file=path, corpus_format="custom")
        result = load_corpus(data)
        parsed = json.loads(result)
        assert parsed == payload
    finally:
        os.unlink(path)
