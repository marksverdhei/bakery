"""Tests for ContextConfig dataclass and HfArgumentParser integration."""

from transformers import HfArgumentParser

from bakery.config import BakeryConfig, ContextConfig, DataConfig, LoraConfig


def test_context_config_defaults():
    cfg = ContextConfig()
    assert cfg.prefix_messages is None
    assert cfg.prefix_messages_file is None
    assert cfg.student_retained_turns == 0
    assert cfg.target_roles == ["assistant"]
    assert cfg.target_content_pattern is None


def test_context_config_target_roles_not_shared_across_instances():
    """Mutable default: each instance should get its own list."""
    a = ContextConfig()
    b = ContextConfig()
    a.target_roles.append("user")
    assert b.target_roles == ["assistant"]


def test_context_config_explicit_fields():
    cfg = ContextConfig(
        prefix_messages=[{"role": "system", "content": "s"}],
        student_retained_turns=2,
        target_roles=["assistant", "tool"],
        target_content_pattern=r"^A",
    )
    assert cfg.prefix_messages == [{"role": "system", "content": "s"}]
    assert cfg.student_retained_turns == 2
    assert cfg.target_roles == ["assistant", "tool"]
    assert cfg.target_content_pattern == r"^A"


def test_hf_argument_parser_includes_context_fields():
    """ContextConfig fields are exposed as top-level CLI flags via HfArgumentParser."""
    parser = HfArgumentParser((BakeryConfig, DataConfig, LoraConfig, ContextConfig))
    baking, data, lora, context = parser.parse_args_into_dataclasses(
        args=[
            "--output_dir",
            "/tmp/hftest",
            "--model_name_or_path",
            "gpt2",
            "--student_retained_turns",
            "3",
            "--target_content_pattern",
            r"^Answer:",
        ]
    )
    assert context.student_retained_turns == 3
    assert context.target_content_pattern == r"^Answer:"
    assert context.target_roles == ["assistant"]  # default preserved


def test_bakery_config_system_prompt_still_accepted():
    """Deprecated system_prompt field still parseable for back-compat."""
    parser = HfArgumentParser((BakeryConfig, DataConfig, LoraConfig, ContextConfig))
    baking, _, _, _ = parser.parse_args_into_dataclasses(
        args=[
            "--output_dir",
            "/tmp/bp_test",
            "--model_name_or_path",
            "gpt2",
            "--system_prompt",
            "You are helpful.",
        ]
    )
    assert baking.system_prompt == "You are helpful."


def test_bakery_config_and_context_config_coexist():
    """Both old system_prompt and new prefix_messages fields can be set simultaneously
    — CLI layer is responsible for reconciling / warning."""
    parser = HfArgumentParser((BakeryConfig, DataConfig, LoraConfig, ContextConfig))
    baking, _, _, context = parser.parse_args_into_dataclasses(
        args=[
            "--output_dir",
            "/tmp/both",
            "--model_name_or_path",
            "gpt2",
            "--system_prompt",
            "old-style prompt",
            "--student_retained_turns",
            "1",
        ]
    )
    assert baking.system_prompt == "old-style prompt"
    assert context.student_retained_turns == 1
