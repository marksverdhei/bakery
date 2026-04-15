"""Tests for CLI helper functions (_load_prefix_file)."""

import json
import os
import tempfile

import pytest

from bakery.cli import _load_prefix_file


def _write(suffix: str, content: str) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
    f.write(content)
    f.close()
    return f.name


def test_load_prefix_file_json():
    path = _write(
        ".json",
        json.dumps(
            [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
            ]
        ),
    )
    try:
        result = _load_prefix_file(path)
        assert result == [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
        ]
    finally:
        os.unlink(path)


def test_load_prefix_file_yaml():
    path = _write(
        ".yaml",
        "- role: system\n  content: s\n- role: user\n  content: u\n",
    )
    try:
        result = _load_prefix_file(path)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["content"] == "u"
    finally:
        os.unlink(path)


def test_load_prefix_file_yml_extension():
    path = _write(
        ".yml",
        "- role: system\n  content: hi\n",
    )
    try:
        result = _load_prefix_file(path)
        assert result == [{"role": "system", "content": "hi"}]
    finally:
        os.unlink(path)


def test_load_prefix_file_rejects_non_list_dict():
    path = _write(".json", json.dumps({"role": "system", "content": "s"}))
    try:
        with pytest.raises(ValueError, match="must contain a JSON/YAML list"):
            _load_prefix_file(path)
    finally:
        os.unlink(path)


def test_load_prefix_file_rejects_non_list_string():
    path = _write(".json", json.dumps("just a string"))
    try:
        with pytest.raises(ValueError):
            _load_prefix_file(path)
    finally:
        os.unlink(path)


def test_load_prefix_file_empty_list():
    """An empty list is technically valid — caller may want to reject later."""
    path = _write(".json", json.dumps([]))
    try:
        result = _load_prefix_file(path)
        assert result == []
    finally:
        os.unlink(path)


def test_load_prefix_file_missing_file():
    with pytest.raises(FileNotFoundError):
        _load_prefix_file("/nonexistent/path/prefix.json")
