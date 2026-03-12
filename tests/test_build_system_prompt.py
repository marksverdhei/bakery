import pytest

from bakery.config import BakeryConfig, DataConfig
from bakery.data import build_system_prompt


def test_explicit_system_prompt():
    baking = BakeryConfig(output_dir="/tmp/test", system_prompt="You are helpful.")
    data = DataConfig()
    assert build_system_prompt(baking, data) == "You are helpful."


def test_template_with_corpus():
    baking = BakeryConfig(output_dir="/tmp/test")
    data = DataConfig(system_prompt_template="Knowledge: {corpus}")
    result = build_system_prompt(baking, data, corpus="some facts")
    assert result == "Knowledge: some facts"


def test_corpus_default_template():
    baking = BakeryConfig(output_dir="/tmp/test")
    data = DataConfig()
    result = build_system_prompt(baking, data, corpus="some facts")
    assert "some facts" in result
    assert "knowledge base" in result.lower()


def test_no_prompt_raises():
    baking = BakeryConfig(output_dir="/tmp/test")
    data = DataConfig()
    with pytest.raises(ValueError, match="No system prompt configured"):
        build_system_prompt(baking, data)
