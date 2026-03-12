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
