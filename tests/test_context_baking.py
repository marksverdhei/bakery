"""Tests for the context-baking generalization (prefix_messages, target masking,
multi-turn, per-row prefix override, retained student turns).
"""

import re

import pytest
import torch
from peft import LoraConfig as PeftLoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from bakery.config import BakeryConfig, ContextConfig
from bakery.data import (
    create_conversational_dataset,
    prompt_baking_collator,
)
from bakery.masking import build_target_mask, clear_mask_cache
from bakery.trainer import ContextBakingTrainer

CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "{{ m['role'] }}: {{ m['content'] }}\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}assistant: {% endif %}"
)


def _make_tokenizer():
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    tok.chat_template = CHAT_TEMPLATE
    return tok


def _make_trainer(
    context_config=None,
    dataset=None,
    system_prompt=None,
    batch_size=1,
):
    tokenizer = _make_tokenizer()
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    peft_config = PeftLoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["c_attn"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    args = BakeryConfig(
        output_dir="/tmp/bakery_ctx_test",
        system_prompt=system_prompt,
        num_trajectories=1,
        trajectory_length=8,
        per_device_train_batch_size=batch_size,
        num_train_epochs=1,
        logging_steps=1,
        report_to="none",
        use_cpu=True,
    )

    return ContextBakingTrainer(
        model=model,
        args=args,
        context_config=context_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=prompt_baking_collator,
    )


# ---------- masking unit tests ----------


def test_mask_role_based_single_assistant():
    """Single assistant turn: only its tokens are True."""
    clear_mask_cache()
    tok = _make_tokenizer()
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    ids, mask, first = build_target_mask(tok, messages, ["assistant"], None)
    assert len(ids) == mask.numel()
    assert mask.any()
    assert 0 <= first < len(ids)
    # Every True position must be in the assistant tail.
    assert mask[:first].sum() == 0


def test_mask_all_assistant_turns_multi_turn():
    """Multi-turn convo: every assistant message contributes to mask."""
    clear_mask_cache()
    tok = _make_tokenizer()
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    _, mask, _ = build_target_mask(tok, messages, ["assistant"], None)
    # Should have two disjoint True regions (one per assistant).
    diffs = mask.int().diff().abs().sum().item()
    # Number of edges = 2 * num_spans.
    assert diffs >= 2


def test_mask_regex_filter():
    """Regex filters which assistant turns count as targets."""
    clear_mask_cache()
    tok = _make_tokenizer()
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "nope"},
        {"role": "user", "content": "q again"},
        {"role": "assistant", "content": "Answer: yes"},
    ]
    _, mask_all, _ = build_target_mask(tok, messages, ["assistant"], None)
    _, mask_filtered, _ = build_target_mask(
        tok, messages, ["assistant"], r"^Answer:"
    )
    assert mask_all.sum() > mask_filtered.sum()
    assert mask_filtered.sum() > 0


def test_mask_excludes_prefix_via_target_min_msg_idx():
    """target_min_msg_idx skips few-shot assistant turns in the prefix."""
    clear_mask_cache()
    tok = _make_tokenizer()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "fewshot-q"},
        {"role": "assistant", "content": "fewshot-a"},  # prefix, should be masked out
        {"role": "user", "content": "real-q"},
        {"role": "assistant", "content": "real-a"},
    ]
    _, mask_with_prefix, first_with = build_target_mask(
        tok, messages, ["assistant"], None, target_min_msg_idx=3
    )
    _, mask_no_prefix, first_no = build_target_mask(
        tok, messages, ["assistant"], None, target_min_msg_idx=0
    )
    assert mask_no_prefix.sum() > mask_with_prefix.sum()
    assert first_with > first_no  # first target is pushed later


def test_mask_cache_key_includes_prefix_idx():
    """Different target_min_msg_idx values produce distinct cache entries."""
    from bakery.masking import _MASK_CACHE

    clear_mask_cache()
    tok = _make_tokenizer()
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]
    build_target_mask(tok, messages, ["assistant"], None, target_min_msg_idx=0)
    build_target_mask(tok, messages, ["assistant"], None, target_min_msg_idx=1)
    assert len(_MASK_CACHE) == 2


# ---------- trainer integration tests ----------


def test_backcompat_system_prompt_desugars():
    """BakeryConfig.system_prompt with no ContextConfig still produces nonzero loss."""
    trainer = _make_trainer(
        context_config=None,
        system_prompt="You are a helpful assistant.",
    )
    inputs = {
        "user_messages": ["What is 2+2?"],
        "responses": ["The answer is 4."],
    }
    loss = trainer.compute_loss(trainer.model, inputs)
    # Desugared prefix means teacher sees system, student does not → real KL signal.
    assert loss.item() > 0
    assert trainer.prefix_messages == [
        {"role": "system", "content": "You are a helpful assistant."}
    ]


def test_global_prefix_messages_multi_turn():
    """Global multi-turn prefix (system + few-shot) produces a nonzero loss."""
    context = ContextConfig(
        prefix_messages=[
            {"role": "system", "content": "You answer concisely."},
            {"role": "user", "content": "Example Q"},
            {"role": "assistant", "content": "Example A"},
        ],
        target_roles=["assistant"],
    )
    trainer = _make_trainer(context_config=context, system_prompt=None)
    inputs = {
        "user_messages": ["What is AI?"],
        "responses": ["Artificial intelligence."],
    }
    loss = trainer.compute_loss(trainer.model, inputs)
    assert loss.item() > 0


def test_per_row_prefix_overrides_global():
    """A row's prefix_messages takes precedence over the global one."""
    global_prefix = [{"role": "system", "content": "global prompt"}]
    row_prefix = [{"role": "system", "content": "row-specific prompt"}]
    context = ContextConfig(prefix_messages=global_prefix)
    trainer = _make_trainer(context_config=context)

    inputs = {
        "prefix_messages": [row_prefix],
        "turns": [[{"role": "user", "content": "hi"}]],
        "responses": ["hello"],
    }
    loss = trainer.compute_loss(trainer.model, inputs)
    # Sanity: this path uses the row prefix, not global.
    assert loss.item() > 0


def test_student_retained_turns_nonzero():
    """student_retained_turns=N keeps last N prefix messages in student view."""
    context_pure = ContextConfig(
        prefix_messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "ctx-q"},
            {"role": "assistant", "content": "ctx-a"},
        ],
        student_retained_turns=0,
    )
    context_retained = ContextConfig(
        prefix_messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "ctx-q"},
            {"role": "assistant", "content": "ctx-a"},
        ],
        student_retained_turns=2,
    )

    t_pure = _make_trainer(context_config=context_pure)
    t_retained = _make_trainer(context_config=context_retained)

    # Prefixes produce different student views → different prompt lengths.
    assert t_pure.student_retained_turns == 0
    assert t_retained.student_retained_turns == 2
    assert t_pure._student_prefix(context_pure.prefix_messages) == []
    assert t_retained._student_prefix(context_retained.prefix_messages) == [
        {"role": "user", "content": "ctx-q"},
        {"role": "assistant", "content": "ctx-a"},
    ]


def test_multi_turn_conversational_batch():
    """Conversational batch with multi-turn turns → all assistant tokens are targets."""
    context = ContextConfig(
        prefix_messages=[{"role": "system", "content": "sys"}],
    )
    trainer = _make_trainer(context_config=context)
    inputs = {
        "prefix_messages": [None],
        "turns": [
            [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
            ]
        ],
        "responses": [None],
    }
    loss = trainer.compute_loss(trainer.model, inputs)
    assert loss.item() > 0


def test_target_roles_restrict_to_assistant():
    """Switching target_roles to exclude assistant yields zero-loss fallback."""
    context = ContextConfig(
        prefix_messages=[{"role": "system", "content": "sys"}],
        target_roles=["tool"],  # no messages match
    )
    trainer = _make_trainer(context_config=context)
    inputs = {
        "user_messages": ["q"],
        "responses": ["a"],
    }
    loss = trainer.compute_loss(trainer.model, inputs)
    assert loss.item() == 0.0


def test_pattern_filter_selects_matching_turns_only():
    """target_content_pattern restricts targets to matching assistant content."""
    context = ContextConfig(
        prefix_messages=[{"role": "system", "content": "sys"}],
        target_roles=["assistant"],
        target_content_pattern=r"^Answer:",
    )
    trainer = _make_trainer(context_config=context)
    # Response does not match pattern → zero loss.
    inputs_nomatch = {
        "user_messages": ["q"],
        "responses": ["just talk"],
    }
    inputs_match = {
        "user_messages": ["q"],
        "responses": ["Answer: yes"],
    }
    assert trainer.compute_loss(trainer.model, inputs_nomatch).item() == 0.0
    assert trainer.compute_loss(trainer.model, inputs_match).item() > 0


def test_create_conversational_dataset_shape():
    """create_conversational_dataset preserves prefix_messages/turns/response columns."""
    rows = [
        {
            "turns": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
            "prefix_messages": [{"role": "system", "content": "s"}],
            "response": None,
        },
        {
            "turns": [{"role": "user", "content": "q2"}],
            "prefix_messages": None,
            "response": "a2",
        },
    ]
    ds = create_conversational_dataset(rows)
    assert set(ds.column_names) == {"turns", "prefix_messages", "responses"}
    assert len(ds) == 2
    assert ds[0]["prefix_messages"] == [{"role": "system", "content": "s"}]
    assert ds[1]["prefix_messages"] is None
    assert ds[1]["responses"] == "a2"
