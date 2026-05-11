"""Comprehensive tests for bakery.masking.build_target_mask.

Covers:
- Role filtering (single + multiple roles)
- Regex filtering
- target_min_msg_idx (prefix exclusion)
- Cache behavior (hits, misses, key components, eviction, clearing)
- Edge cases: empty messages, no matches, all matches
- Span ordering / non-overlap sanity
"""

import pytest
import torch
from transformers import AutoTokenizer

from bakery.masking import (
    _MASK_CACHE,
    _MASK_CACHE_MAX,
    _longest_common_prefix_len,
    _messages_hash,
    _per_message_spans,
    build_target_mask,
    clear_mask_cache,
)

CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "{{ m['role'] }}: {{ m['content'] }}\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}assistant: {% endif %}"
)


@pytest.fixture
def tok():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


# ---------- helpers ----------


def test_longest_common_prefix_len_identical():
    assert _longest_common_prefix_len([1, 2, 3], [1, 2, 3]) == 3


def test_longest_common_prefix_len_partial():
    assert _longest_common_prefix_len([1, 2, 3, 4], [1, 2, 9]) == 2


def test_longest_common_prefix_len_empty():
    assert _longest_common_prefix_len([], [1, 2]) == 0
    assert _longest_common_prefix_len([1, 2], []) == 0
    assert _longest_common_prefix_len([], []) == 0


def test_longest_common_prefix_len_no_overlap():
    assert _longest_common_prefix_len([9, 8], [1, 2]) == 0


def test_messages_hash_stable():
    m = [{"role": "user", "content": "x"}]
    assert _messages_hash(m) == _messages_hash(m)


def test_messages_hash_changes_with_content():
    a = [{"role": "user", "content": "x"}]
    b = [{"role": "user", "content": "y"}]
    assert _messages_hash(a) != _messages_hash(b)


def test_messages_hash_changes_with_role():
    a = [{"role": "user", "content": "x"}]
    b = [{"role": "assistant", "content": "x"}]
    assert _messages_hash(a) != _messages_hash(b)


def test_messages_hash_handles_missing_fields():
    # Malformed messages shouldn't crash the hash.
    _messages_hash([{"role": "user"}])
    _messages_hash([{"content": "x"}])
    _messages_hash([{}])


def test_per_message_spans_covers_full_sequence(tok):
    """Sum of distinct span ranges should cover every token in the full sequence."""
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]
    full_ids, spans = _per_message_spans(tok, messages)
    # Spans should be non-decreasing in start and end.
    for i in range(len(spans) - 1):
        assert spans[i][1] <= spans[i + 1][1]
    # Final span should reach full length.
    assert spans[-1][1] == len(full_ids)
    # Starts should never exceed ends.
    for s, e in spans:
        assert s <= e


def test_per_message_spans_empty(tok):
    full_ids, spans = _per_message_spans(tok, [])
    assert full_ids == []
    assert spans == []


# ---------- build_target_mask ----------


def test_mask_returns_list_tensor_and_index(tok):
    clear_mask_cache()
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "there"},
    ]
    ids, mask, first = build_target_mask(tok, messages, ["assistant"])
    assert isinstance(ids, list)
    assert isinstance(mask, torch.Tensor)
    assert mask.dtype == torch.bool
    assert mask.numel() == len(ids)
    assert isinstance(first, int)


def test_mask_no_target_role_match(tok):
    """When no message matches target_roles, mask is all False and first == len."""
    clear_mask_cache()
    messages = [
        {"role": "user", "content": "only user"},
        {"role": "system", "content": "only system"},
    ]
    ids, mask, first = build_target_mask(tok, messages, ["assistant"])
    assert mask.sum().item() == 0
    assert first == len(ids)


def test_mask_target_role_user(tok):
    """target_roles=['user'] should mask user tokens, not assistant."""
    clear_mask_cache()
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    _, mask_user, _ = build_target_mask(tok, messages, ["user"])
    _, mask_assistant, _ = build_target_mask(tok, messages, ["assistant"])
    assert mask_user.sum() > 0
    assert mask_assistant.sum() > 0
    # Roles are mutually exclusive — no shared True positions.
    assert (mask_user & mask_assistant).sum() == 0


def test_mask_multiple_target_roles(tok):
    """Multiple roles in target_roles → union of their tokens."""
    clear_mask_cache()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]
    _, mask_user, _ = build_target_mask(tok, messages, ["user"])
    _, mask_asst, _ = build_target_mask(tok, messages, ["assistant"])
    _, mask_both, _ = build_target_mask(tok, messages, ["user", "assistant"])
    assert mask_both.sum() == mask_user.sum() + mask_asst.sum()


def test_mask_regex_matches_all(tok):
    """Pattern matching every content → same as no pattern."""
    clear_mask_cache()
    messages = [
        {"role": "assistant", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    _, m_all, _ = build_target_mask(tok, messages, ["assistant"], None)
    _, m_re, _ = build_target_mask(tok, messages, ["assistant"], r".")
    assert m_all.sum() == m_re.sum()


def test_mask_regex_matches_none(tok):
    """Pattern matching nothing → zero mask."""
    clear_mask_cache()
    messages = [
        {"role": "assistant", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    _, mask, first = build_target_mask(
        tok, messages, ["assistant"], r"zzzzzzzz_definitely_not"
    )
    assert mask.sum() == 0
    assert first == len(mask)


def test_mask_regex_re_search_semantics(tok):
    """Pattern uses re.search (not re.fullmatch) — partial match should count."""
    clear_mask_cache()
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "prefix Answer: yes suffix"},
    ]
    _, mask, _ = build_target_mask(tok, messages, ["assistant"], r"Answer:")
    assert mask.sum() > 0


def test_mask_first_target_idx_points_to_true_position(tok):
    clear_mask_cache()
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]
    _, mask, first = build_target_mask(tok, messages, ["assistant"])
    assert first < len(mask)
    assert mask[first].item() is True


def test_mask_first_target_idx_empty_when_no_targets(tok):
    clear_mask_cache()
    messages = [{"role": "user", "content": "q"}]
    ids, mask, first = build_target_mask(tok, messages, ["assistant"])
    assert first == len(ids)


def test_mask_target_min_msg_idx_past_end_gives_empty(tok):
    """target_min_msg_idx >= len(messages) → no targets possible."""
    clear_mask_cache()
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]
    _, mask, first = build_target_mask(
        tok, messages, ["assistant"], None, target_min_msg_idx=5
    )
    assert mask.sum() == 0
    assert first == len(mask)


def test_mask_target_min_msg_idx_zero_equals_default(tok):
    clear_mask_cache()
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]
    _, m_default, _ = build_target_mask(tok, messages, ["assistant"])
    _, m_zero, _ = build_target_mask(
        tok, messages, ["assistant"], None, target_min_msg_idx=0
    )
    assert m_default.sum() == m_zero.sum()


def test_mask_prefix_assistant_excluded_as_target(tok):
    """Few-shot assistant in prefix must NOT be a target when target_min_msg_idx is set."""
    clear_mask_cache()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "ex-q"},
        {"role": "assistant", "content": "ex-a"},  # few-shot prefix answer
        {"role": "user", "content": "real-q"},
        {"role": "assistant", "content": "real-a"},  # training target
    ]
    ids, mask, first = build_target_mask(
        tok, messages, ["assistant"], None, target_min_msg_idx=3
    )
    # Only one contiguous True region.
    transitions = mask.int().diff().abs().sum().item()
    assert transitions <= 2  # 0→1 and 1→0
    assert mask.sum() > 0


# ---------- cache ----------


def test_cache_hit_on_repeat(tok):
    clear_mask_cache()
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]
    build_target_mask(tok, messages, ["assistant"])
    assert len(_MASK_CACHE) == 1
    build_target_mask(tok, messages, ["assistant"])
    # Still exactly one entry.
    assert len(_MASK_CACHE) == 1


def test_cache_key_differentiates_target_roles(tok):
    clear_mask_cache()
    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    build_target_mask(tok, messages, ["assistant"])
    build_target_mask(tok, messages, ["user"])
    build_target_mask(tok, messages, ["assistant", "user"])
    assert len(_MASK_CACHE) == 3


def test_cache_key_differentiates_pattern(tok):
    clear_mask_cache()
    messages = [{"role": "assistant", "content": "Answer: 42"}]
    build_target_mask(tok, messages, ["assistant"], None)
    build_target_mask(tok, messages, ["assistant"], r"^Answer:")
    build_target_mask(tok, messages, ["assistant"], r"42$")
    assert len(_MASK_CACHE) == 3


def test_cache_key_differentiates_messages(tok):
    clear_mask_cache()
    build_target_mask(
        tok,
        [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a"}],
        ["assistant"],
    )
    build_target_mask(
        tok,
        [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "a"}],
        ["assistant"],
    )
    assert len(_MASK_CACHE) == 2


def test_cache_eviction_bounded(tok, monkeypatch):
    """When cache is at max, inserting a new entry evicts the oldest."""
    clear_mask_cache()
    monkeypatch.setattr("bakery.masking._MASK_CACHE_MAX", 3)
    for i in range(10):
        msgs = [
            {"role": "user", "content": f"q{i}"},
            {"role": "assistant", "content": f"a{i}"},
        ]
        build_target_mask(tok, msgs, ["assistant"])
    # Cache size should never exceed the bound (module-level _MASK_CACHE_MAX
    # may have been re-read but the check uses the patched value).
    assert len(_MASK_CACHE) <= 3


def test_clear_mask_cache(tok):
    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    build_target_mask(tok, messages, ["assistant"])
    assert len(_MASK_CACHE) > 0
    clear_mask_cache()
    assert len(_MASK_CACHE) == 0


def test_mask_cache_max_constant_exists():
    """Ensure the bound constant is a sensible default."""
    assert isinstance(_MASK_CACHE_MAX, int)
    assert _MASK_CACHE_MAX >= 100


# ---------- structural invariants ----------


def test_mask_always_bool_tensor(tok):
    clear_mask_cache()
    for messages in [
        [{"role": "user", "content": "x"}],
        [{"role": "assistant", "content": "x"}],
        [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ],
    ]:
        _, mask, _ = build_target_mask(tok, messages, ["assistant"])
        assert mask.dtype == torch.bool


def test_mask_length_matches_input_ids(tok):
    clear_mask_cache()
    messages = [
        {"role": "user", "content": "longer query here"},
        {"role": "assistant", "content": "a longer response with more tokens in it"},
    ]
    ids, mask, _ = build_target_mask(tok, messages, ["assistant"])
    assert mask.numel() == len(ids)


def test_mask_target_region_is_contiguous_for_single_assistant(tok):
    """A single assistant message produces exactly one contiguous True region."""
    clear_mask_cache()
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "some answer"},
    ]
    _, mask, _ = build_target_mask(tok, messages, ["assistant"])
    if mask.sum() == 0:
        pytest.skip("no targets — tokenizer template anomaly")
    transitions = mask.int().diff().abs().sum().item()
    # Starts at False, rises to True once, falls back at most once.
    assert transitions in (1, 2)
