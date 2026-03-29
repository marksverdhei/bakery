import sys

import pytest

from bakery.config import BakeryConfig, DataConfig, LoraConfig

_HF_PARSER_BROKEN = sys.version_info >= (3, 14)


def _make_hf_parser():
    if _HF_PARSER_BROKEN:
        pytest.skip(
            "HfArgumentParser incompatible with Python 3.14+ (transformers bug)"
        )
    from transformers import HfArgumentParser

    return HfArgumentParser((BakeryConfig, DataConfig, LoraConfig))


def test_parser_accepts_flat_args():
    parser = _make_hf_parser()
    baking, data, lora = parser.parse_args_into_dataclasses(
        args=[
            "--output_dir",
            "/tmp/test",
            "--system_prompt",
            "You are helpful.",
            "--model_name_or_path",
            "gpt2",
            "--r",
            "16",
        ]
    )
    assert baking.system_prompt == "You are helpful."
    assert data.model_name_or_path == "gpt2"
    assert lora.r == 16
    assert baking.remove_unused_columns is False


def test_default_lora_targets():
    lora = LoraConfig()
    assert "q_proj" in lora.target_modules
    assert len(lora.target_modules) == 7


# ---------------------------------------------------------------------------
# LoraConfig.__post_init__ normalisation — target_modules: all / all-linear
# ---------------------------------------------------------------------------


def test_lora_target_modules_all_string_normalised():
    """YAML scalar 'all' should become PEFT's 'all-linear' string."""
    lora = LoraConfig(target_modules="all")
    assert lora.target_modules == "all-linear"


def test_lora_target_modules_all_list_normalised():
    """YAML list ['all'] should become the string 'all-linear'."""
    lora = LoraConfig(target_modules=["all"])
    assert lora.target_modules == "all-linear"


def test_lora_target_modules_all_linear_list_normalised():
    """['all-linear'] list should be unwrapped to the bare string."""
    lora = LoraConfig(target_modules=["all-linear"])
    assert lora.target_modules == "all-linear"


def test_lora_target_modules_all_linear_string_unchanged():
    """Passing 'all-linear' string directly should be kept as-is."""
    lora = LoraConfig(target_modules="all-linear")
    assert lora.target_modules == "all-linear"


def test_lora_target_modules_explicit_list_unchanged():
    """An explicit list of module names should not be normalised."""
    modules = ["q_proj", "v_proj"]
    lora = LoraConfig(target_modules=modules)
    assert lora.target_modules == modules


# ---------------------------------------------------------------------------
# DataConfig — eval_dataset (separate HF dataset for validation loss)
# ---------------------------------------------------------------------------


def test_eval_dataset_defaults_to_none():
    data = DataConfig.__new__(DataConfig)
    data.__init__()
    assert data.eval_dataset is None


def test_eval_dataset_set():
    data = DataConfig(eval_dataset="HuggingFaceH4/ultrachat_200k")
    assert data.eval_dataset == "HuggingFaceH4/ultrachat_200k"


def test_eval_dataset_independent_of_eval_split():
    """eval_dataset and eval_dataset_split are independent — both can be set."""
    data = DataConfig(
        eval_dataset="myorg/myrepo",
        eval_dataset_split="test_sft[:100]",
    )
    assert data.eval_dataset == "myorg/myrepo"
    assert data.eval_dataset_split == "test_sft[:100]"


def test_eval_dataset_accepted_by_parser():
    parser = _make_hf_parser()
    _, data, _ = parser.parse_args_into_dataclasses(
        args=[
            "--output_dir",
            "/tmp/t",
            "--eval_dataset",
            "HuggingFaceH4/ultrachat_200k",
            "--eval_dataset_split",
            "test_sft[:50]",
        ]
    )
    assert data.eval_dataset == "HuggingFaceH4/ultrachat_200k"
    assert data.eval_dataset_split == "test_sft[:50]"


def test_bakery_config_numeric_coercion():
    """Numeric BakeryConfig fields should be coerced from strings."""
    # HfArgumentParser may pass floats as strings from YAML
    config = BakeryConfig(
        output_dir="/tmp/test",
        learning_rate="1e-4",
        temperature="0.8",
        num_trajectories="4",
        trajectory_length="128",
    )
    assert isinstance(config.learning_rate, float)
    assert abs(config.learning_rate - 1e-4) < 1e-10
    assert isinstance(config.temperature, float)
    assert config.num_trajectories == 4
    assert config.trajectory_length == 128
