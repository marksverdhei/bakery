"""Target-mask construction for context baking.

Given a list of chat messages, produce a per-token boolean mask indicating
which tokens are training targets (i.e. tokens that should receive KL loss).
Selection is controlled by `target_roles` (message role filter) and an optional
`target_content_pattern` (regex on message content).

Strategy: tokenize incremental prefixes of the message list and derive per-message
token spans by diffing cumulative token counts. For each message that matches the
target criteria, its span in the final tokenized sequence is marked True.

This approach is tokenizer- and chat-template-agnostic. It assumes cumulative
`apply_chat_template(messages[:k], tokenize=True)` produces a token-prefix of
`apply_chat_template(messages[:k+1], tokenize=True)`. This holds for well-formed
chat templates (Llama, Qwen, Gemma, Mistral). A longest-common-token-prefix
fallback handles templates that close blocks at the end of each render.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import torch


def _tokenize_prefix(tokenizer, messages: List[dict]) -> List[int]:
    """Render and tokenize a message-list prefix; return input_ids as a list."""
    if not messages:
        return []
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    ids = tokenizer(rendered, add_special_tokens=False, return_tensors=None)["input_ids"]
    # Some tokenizers return List[List[int]] when given a single string; flatten.
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def _longest_common_prefix_len(a: List[int], b: List[int]) -> int:
    """Length of the longest shared token prefix between two id sequences."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _per_message_spans(
    tokenizer, messages: List[dict]
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Return (full_input_ids, spans) where spans[i] is (start, end) of messages[i]
    in the full tokenized sequence.

    Uses longest-common-token-prefix to be resilient to templates that close
    blocks at each render (the closing tokens of messages[:k] won't appear in
    messages[:k+1], so we take the common prefix as the "fixed" boundary).
    """
    full = _tokenize_prefix(tokenizer, messages)
    spans: List[Tuple[int, int]] = []
    prev_boundary = 0
    prev_tokens: List[int] = []
    for i in range(len(messages)):
        this_tokens = _tokenize_prefix(tokenizer, messages[: i + 1])
        # Boundary between message i-1 and i is the longest common prefix
        # between prev_tokens (messages[:i]) and this_tokens (messages[:i+1]).
        if i == 0:
            start = 0
        else:
            start = _longest_common_prefix_len(prev_tokens, this_tokens)
            # Clamp to the previous boundary: each message can only extend forward.
            start = max(start, prev_boundary)
        # End of message i = longest common prefix between this_tokens and full.
        end = _longest_common_prefix_len(this_tokens, full)
        end = max(end, start)
        spans.append((start, end))
        prev_boundary = end
        prev_tokens = this_tokens
    return full, spans


def _messages_hash(messages: List[dict]) -> str:
    """Stable content hash of a message list for caching."""
    import hashlib

    h = hashlib.blake2b(digest_size=16)
    for m in messages:
        h.update(str(m.get("role", "")).encode())
        h.update(b"\x1f")
        h.update(str(m.get("content", "")).encode())
        h.update(b"\x1e")
    return h.hexdigest()


# Bounded cache; tokenizer identity is part of the key via id().
_MASK_CACHE: "dict[tuple, tuple[list[int], torch.BoolTensor, int]]" = {}
_MASK_CACHE_MAX = 10_000


def build_target_mask(
    tokenizer,
    messages: List[dict],
    target_roles: List[str],
    content_pattern: Optional[str] = None,
    target_min_msg_idx: int = 0,
) -> Tuple[List[int], torch.BoolTensor, int]:
    """Compute per-token target mask for a message list.

    Args:
        tokenizer: HF tokenizer with `apply_chat_template` support.
        messages: list of {role, content} dicts (full sequence including targets).
        target_roles: roles whose tokens should be KL targets.
        content_pattern: optional regex; if set, message must also match via re.search.
        target_min_msg_idx: messages at indices < this are never targets (used to
            exclude the prefix portion when it happens to contain assistant turns
            like few-shot examples that should be baked, not trained on).

    Returns:
        (input_ids, mask, first_target_idx) where:
          - input_ids: tokenized full sequence (list[int])
          - mask: BoolTensor of shape (len(input_ids),), True for target tokens
          - first_target_idx: index of the first True token (or len(input_ids) if none)
    """
    roles_tuple = tuple(target_roles)
    cache_key = (
        id(tokenizer),
        _messages_hash(messages),
        roles_tuple,
        content_pattern,
        target_min_msg_idx,
    )
    cached = _MASK_CACHE.get(cache_key)
    if cached is not None:
        return cached

    pattern = re.compile(content_pattern) if content_pattern else None
    role_set = set(target_roles)

    input_ids, spans = _per_message_spans(tokenizer, messages)
    mask = torch.zeros(len(input_ids), dtype=torch.bool)
    first_target_idx = len(input_ids)

    for i, msg in enumerate(messages):
        if i < target_min_msg_idx:
            continue
        if msg.get("role") not in role_set:
            continue
        if pattern is not None and not pattern.search(str(msg.get("content", ""))):
            continue
        start, end = spans[i]
        if end > start:
            mask[start:end] = True
            if start < first_target_idx:
                first_target_idx = start

    # Bound the cache.
    if len(_MASK_CACHE) >= _MASK_CACHE_MAX:
        # Drop an arbitrary entry (dict preserves insertion order; pop oldest).
        _MASK_CACHE.pop(next(iter(_MASK_CACHE)))
    _MASK_CACHE[cache_key] = (input_ids, mask, first_target_idx)

    return input_ids, mask, first_target_idx


def clear_mask_cache() -> None:
    """Clear the target-mask cache (useful in tests)."""
    _MASK_CACHE.clear()
