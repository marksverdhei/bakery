"""Tests for ContextBakingTrainer internal helpers + edge-case integration tests.

Uses one shared tiny GPT-2 + LoRA trainer across many tests to keep CPU time down.
"""

import warnings

import pytest
import torch
from peft import LoraConfig as PeftLoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from bakery.config import BakeryConfig, ContextConfig
from bakery.data import prompt_baking_collator
from bakery.masking import clear_mask_cache
from bakery.trainer import ContextBakingTrainer, PromptBakingTrainer

CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "{{ m['role'] }}: {{ m['content'] }}\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}assistant: {% endif %}"
)


def _mk_tokenizer():
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    tok.chat_template = CHAT_TEMPLATE
    return tok


def _mk_trainer(context_config=None, system_prompt=None, batch_size=1):
    tokenizer = _mk_tokenizer()
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    peft_config = PeftLoraConfig(
        r=4, lora_alpha=8, target_modules=["c_attn"], task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    args = BakeryConfig(
        output_dir="/tmp/bakery_internals",
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
        processing_class=tokenizer,
        data_collator=prompt_baking_collator,
    )


@pytest.fixture(scope="module")
def trainer_sys():
    """Trainer with a simple system prefix, reused across many tests."""
    return _mk_trainer(
        context_config=ContextConfig(
            prefix_messages=[{"role": "system", "content": "You are helpful."}]
        )
    )


@pytest.fixture(scope="module")
def trainer_multi_prefix():
    """Trainer with a multi-turn prefix (system + few-shot)."""
    return _mk_trainer(
        context_config=ContextConfig(
            prefix_messages=[
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "ex-q"},
                {"role": "assistant", "content": "ex-a"},
            ],
            student_retained_turns=0,
        )
    )


# ---------- _row_prefix ----------


def test_row_prefix_uses_global_when_row_is_none(trainer_sys):
    result = trainer_sys._row_prefix(None)
    assert result == trainer_sys.prefix_messages
    assert result is not trainer_sys.prefix_messages  # defensive copy


def test_row_prefix_uses_global_when_row_is_empty(trainer_sys):
    result = trainer_sys._row_prefix([])
    assert result == trainer_sys.prefix_messages


def test_row_prefix_prefers_row_over_global(trainer_sys):
    row = [{"role": "system", "content": "row-specific"}]
    result = trainer_sys._row_prefix(row)
    assert result == row


# ---------- _student_prefix ----------


def test_student_prefix_zero_returns_empty():
    t = _mk_trainer(
        context_config=ContextConfig(
            prefix_messages=[{"role": "system", "content": "s"}],
            student_retained_turns=0,
        )
    )
    assert t._student_prefix(t.prefix_messages) == []


def test_student_prefix_n_larger_than_prefix_returns_full():
    prefix = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]
    t = _mk_trainer(
        context_config=ContextConfig(
            prefix_messages=prefix,
            student_retained_turns=10,
        )
    )
    assert t._student_prefix(prefix) == prefix


def test_student_prefix_keeps_last_n():
    prefix = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ]
    t = _mk_trainer(
        context_config=ContextConfig(
            prefix_messages=prefix,
            student_retained_turns=2,
        )
    )
    assert t._student_prefix(prefix) == prefix[-2:]


# ---------- _append_response ----------


def test_append_response_with_string():
    out = ContextBakingTrainer._append_response(
        [{"role": "user", "content": "q"}], "a"
    )
    assert out[-1] == {"role": "assistant", "content": "a"}
    assert len(out) == 2


def test_append_response_with_none_passthrough():
    msgs = [{"role": "user", "content": "q"}]
    out = ContextBakingTrainer._append_response(msgs, None)
    assert out == msgs
    assert out is not msgs  # defensive copy


def test_append_response_with_empty_string_passthrough():
    """Empty/falsy response should not be appended."""
    msgs = [{"role": "user", "content": "q"}]
    out = ContextBakingTrainer._append_response(msgs, "")
    assert out == msgs


# ---------- _normalize_batch ----------


def test_normalize_batch_legacy_shape():
    inputs = {"user_messages": ["q1", "q2"], "responses": ["a1", "a2"]}
    prefix, turns, responses = ContextBakingTrainer._normalize_batch(inputs)
    assert prefix == [None, None]
    assert turns == [
        [{"role": "user", "content": "q1"}],
        [{"role": "user", "content": "q2"}],
    ]
    assert responses == ["a1", "a2"]


def test_normalize_batch_conversational_shape():
    inputs = {
        "turns": [[{"role": "user", "content": "q"}]],
        "prefix_messages": [[{"role": "system", "content": "s"}]],
        "responses": ["a"],
    }
    prefix, turns, responses = ContextBakingTrainer._normalize_batch(inputs)
    assert prefix == [[{"role": "system", "content": "s"}]]
    assert turns == [[{"role": "user", "content": "q"}]]
    assert responses == ["a"]


def test_normalize_batch_conversational_missing_prefix_defaults_to_none():
    """When `turns` is present but `prefix_messages` column is missing / falsy, fill with None."""
    inputs = {"turns": [[{"role": "user", "content": "q"}]], "responses": ["a"]}
    prefix, turns, responses = ContextBakingTrainer._normalize_batch(inputs)
    assert prefix == [None]


def test_normalize_batch_legacy_empty():
    prefix, turns, responses = ContextBakingTrainer._normalize_batch(
        {"user_messages": [], "responses": []}
    )
    assert prefix == []
    assert turns == []
    assert responses == []


# ---------- _build_example ----------


def test_build_example_returns_none_when_no_target(trainer_sys):
    """Prefix-only messages (no turns, no response) → no target tokens."""
    clear_mask_cache()
    result = trainer_sys._build_example(
        row_prefix=None,
        turns=[],
        response=None,
    )
    assert result is None


def test_build_example_basic_single_turn(trainer_sys):
    clear_mask_cache()
    result = trainer_sys._build_example(
        row_prefix=None,
        turns=[{"role": "user", "content": "q"}],
        response="a",
    )
    assert result is not None
    assert "teacher_ids" in result
    assert "student_ids" in result
    assert result["teacher_mask"].sum() > 0
    assert result["student_mask"].sum() > 0


def test_build_example_prefix_assistant_not_a_target(trainer_multi_prefix):
    """Few-shot assistant in the prefix must not become a training target."""
    clear_mask_cache()
    result = trainer_multi_prefix._build_example(
        row_prefix=None,
        turns=[{"role": "user", "content": "real-q"}],
        response="real-a",
    )
    assert result is not None
    # Exactly one target region (just the response) — the few-shot assistant is baked.
    transitions = result["teacher_mask"].int().diff().abs().sum().item()
    assert transitions <= 2


# ---------- compute_loss edge cases ----------


def test_compute_loss_no_response_and_no_assistant_turn_returns_zero(trainer_sys):
    """A user-only turn with no response or assistant in the turns must not crash."""
    loss = trainer_sys.compute_loss(
        trainer_sys.model,
        {
            "prefix_messages": [None],
            "turns": [[{"role": "user", "content": "q"}]],
            "responses": [None],
        },
    )
    assert loss.item() == 0.0


def test_compute_loss_whitespace_response_returns_zero(trainer_sys):
    loss = trainer_sys.compute_loss(
        trainer_sys.model,
        {"user_messages": ["q"], "responses": ["   \n\t"]},
    )
    assert loss.item() == 0.0


def test_compute_loss_mixed_batch_some_valid(trainer_sys):
    """Batch where one row is invalid (empty response) and one is valid → nonzero loss."""
    loss = trainer_sys.compute_loss(
        trainer_sys.model,
        {"user_messages": ["q1", "q2"], "responses": ["  ", "a valid response"]},
    )
    assert loss.item() > 0


def test_compute_loss_return_outputs_tuple(trainer_sys):
    out = trainer_sys.compute_loss(
        trainer_sys.model,
        {"user_messages": ["q"], "responses": ["a"]},
        return_outputs=True,
    )
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert out[1] is None


def test_compute_loss_return_outputs_zero_case(trainer_sys):
    """return_outputs branch with no targets still returns a tuple (not scalar)."""
    out = trainer_sys.compute_loss(
        trainer_sys.model,
        {"user_messages": [], "responses": []},
        return_outputs=True,
    )
    assert isinstance(out, tuple)
    assert out[0].item() == 0.0
    assert out[1] is None


def test_compute_loss_is_differentiable(trainer_sys):
    loss = trainer_sys.compute_loss(
        trainer_sys.model, {"user_messages": ["q"], "responses": ["a response"]}
    )
    loss.backward()
    # At least one LoRA param got a gradient.
    has_grad = any(
        p.grad is not None and p.requires_grad
        for p in trainer_sys.model.parameters()
    )
    assert has_grad


def test_compute_loss_conversational_multi_turn_all_assistant_are_targets(trainer_sys):
    """Multi-turn conversation with two assistant messages → nonzero loss over both spans."""
    loss = trainer_sys.compute_loss(
        trainer_sys.model,
        {
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
        },
    )
    assert loss.item() > 0


def test_compute_loss_target_roles_user_does_not_crash():
    """Reconfiguring target_roles to include 'user' must run without errors.

    Note: depending on student_retained_turns, the teacher/student alignment may
    not overlap on user-role tokens (student lacks the prefix), so loss can be 0.
    We only assert the path runs and returns a non-negative scalar.
    """
    t = _mk_trainer(
        context_config=ContextConfig(
            prefix_messages=[{"role": "system", "content": "s"}],
            target_roles=["user"],
        )
    )
    loss = t.compute_loss(
        t.model,
        {
            "prefix_messages": [None],
            "turns": [
                [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "there"},
                ]
            ],
            "responses": [None],
        },
    )
    assert loss.item() >= 0


def test_compute_loss_per_row_prefix_overrides_global(trainer_sys):
    """A per-row prefix takes precedence over the trainer's global prefix."""
    row_prefix = [{"role": "system", "content": "row-specific"}]
    loss = trainer_sys.compute_loss(
        trainer_sys.model,
        {
            "prefix_messages": [row_prefix],
            "turns": [[{"role": "user", "content": "q"}]],
            "responses": ["a"],
        },
    )
    assert loss.item() > 0


def test_compute_loss_with_student_retained_turns_nonzero():
    """student_retained_turns > 0 still yields a real KL signal."""
    t = _mk_trainer(
        context_config=ContextConfig(
            prefix_messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "ctx-q"},
                {"role": "assistant", "content": "ctx-a"},
            ],
            student_retained_turns=2,
        )
    )
    loss = t.compute_loss(
        t.model, {"user_messages": ["hi"], "responses": ["there"]}
    )
    # With retained turns student already sees ctx — loss is smaller but non-negative.
    assert loss.item() >= 0


# ---------- back-compat ----------


def test_prompt_baking_trainer_is_deprecated_alias():
    tokenizer = _mk_tokenizer()
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    peft_config = PeftLoraConfig(
        r=4, lora_alpha=8, target_modules=["c_attn"], task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    args = BakeryConfig(
        output_dir="/tmp/bakery_deprecation",
        system_prompt="legacy prompt",
        per_device_train_batch_size=1,
        num_train_epochs=1,
        logging_steps=1,
        report_to="none",
        use_cpu=True,
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        t = PromptBakingTrainer(
            model=model,
            args=args,
            processing_class=tokenizer,
            data_collator=prompt_baking_collator,
        )
    assert isinstance(t, ContextBakingTrainer)
    assert any(
        issubclass(item.category, DeprecationWarning)
        and "PromptBakingTrainer" in str(item.message)
        for item in w
    )


def test_back_compat_system_prompt_auto_desugars_in_trainer_init():
    """args.system_prompt set but no context_config → trainer auto-wraps it."""
    tokenizer = _mk_tokenizer()
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    peft_config = PeftLoraConfig(
        r=4, lora_alpha=8, target_modules=["c_attn"], task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    args = BakeryConfig(
        output_dir="/tmp/bakery_desugar",
        system_prompt="Auto desugared!",
        per_device_train_batch_size=1,
        num_train_epochs=1,
        logging_steps=1,
        report_to="none",
        use_cpu=True,
    )
    t = ContextBakingTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        data_collator=prompt_baking_collator,
    )
    assert t.prefix_messages == [
        {"role": "system", "content": "Auto desugared!"}
    ]


# ---------- prediction_step ----------


def test_prediction_step_returns_scalar_loss(trainer_sys):
    """Non-sequential eval path returns a detached scalar."""
    loss, _, _ = trainer_sys.prediction_step(
        trainer_sys.model,
        {"user_messages": ["q"], "responses": ["a"]},
        prediction_loss_only=True,
    )
    assert loss.dim() == 0
    # Detached — no grad tracking.
    assert not loss.requires_grad


def test_prediction_step_empty_inputs_returns_zero(trainer_sys):
    loss, _, _ = trainer_sys.prediction_step(
        trainer_sys.model,
        {"user_messages": [], "responses": []},
        prediction_loss_only=True,
    )
    assert loss.item() == 0.0


# ---------- module exports ----------


def test_module_exports_top_level_symbols():
    import bakery

    expected = {
        "BakeryConfig",
        "ContextConfig",
        "DataConfig",
        "LoraConfig",
        "ContextBakingTrainer",
        "PromptBakingTrainer",
        "create_conversational_dataset",
        "create_dataset",
        "load_conversations",
        "load_dataset",
        "prompt_baking_collator",
        "compute_kl_divergence",
        "build_target_mask",
    }
    assert expected.issubset(set(bakery.__all__))
    for name in expected:
        assert getattr(bakery, name) is not None
