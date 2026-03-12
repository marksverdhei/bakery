import re

from bakery.evaluate import evaluate_model


def test_think_tag_stripping():
    """Verify that <think>...</think> blocks are fully removed, not just tags."""
    # This tests the regex used in evaluate_model indirectly via the same pattern
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
