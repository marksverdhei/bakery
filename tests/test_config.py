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
# BakeryConfig.__post_init__ — string coercion
# ---------------------------------------------------------------------------

class TestBakeryConfigPostInit:
    def _base_kwargs(self):
        return dict(
            output_dir="/tmp/test",
            system_prompt="Be helpful.",
        )

    def test_remove_unused_columns_always_false(self):
        cfg = BakeryConfig(**self._base_kwargs())
        assert cfg.remove_unused_columns is False

    @pytest.mark.parametrize("attr,value", [
        ("learning_rate", "1e-4"),
        ("temperature", "0.7"),
        ("sampling_temperature", "0.8"),
        ("warmup_ratio", "0.1"),
        ("max_grad_norm", "1.0"),
    ])
    def test_float_string_coerced_to_float(self, attr, value):
        """String numeric values for float fields are coerced to float."""
        kwargs = self._base_kwargs()
        kwargs[attr] = value
        cfg = BakeryConfig(**kwargs)
        assert isinstance(getattr(cfg, attr), float)
        assert getattr(cfg, attr) == pytest.approx(float(value))

    @pytest.mark.parametrize("attr,value", [
        ("num_trajectories", "4"),
        ("trajectory_length", "128"),
        ("logging_steps", "10"),
        ("seed", "42"),
    ])
    def test_int_string_coerced_to_int(self, attr, value):
        """String numeric values for int fields are coerced to int."""
        kwargs = self._base_kwargs()
        kwargs[attr] = value
        cfg = BakeryConfig(**kwargs)
        assert isinstance(getattr(cfg, attr), int)
        assert getattr(cfg, attr) == int(value)

    def test_invalid_float_string_raises(self):
        kwargs = self._base_kwargs()
        kwargs["learning_rate"] = "not-a-number"
        with pytest.raises(ValueError, match="learning_rate"):
            BakeryConfig(**kwargs)

    def test_invalid_int_string_raises(self):
        kwargs = self._base_kwargs()
        kwargs["num_trajectories"] = "not-an-int"
        with pytest.raises(ValueError, match="num_trajectories"):
            BakeryConfig(**kwargs)

    def test_native_float_unchanged(self):
        """Float values passed as float are not modified."""
        kwargs = self._base_kwargs()
        kwargs["learning_rate"] = 5e-5
        cfg = BakeryConfig(**kwargs)
        assert cfg.learning_rate == pytest.approx(5e-5)

    def test_system_prompt_file_loaded(self, tmp_path):
        """system_prompt is loaded from file when system_prompt is None."""
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("  You are a robot.  ")
        cfg = BakeryConfig(
            output_dir="/tmp/test",
            system_prompt=None,
            system_prompt_file=str(prompt_file),
        )
        assert cfg.system_prompt == "You are a robot."

    def test_system_prompt_takes_priority_over_file(self, tmp_path):
        """Explicit system_prompt takes priority — file is not read."""
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("From file.")
        cfg = BakeryConfig(
            output_dir="/tmp/test",
            system_prompt="Direct prompt.",
            system_prompt_file=str(prompt_file),
        )
        assert cfg.system_prompt == "Direct prompt."


# ---------------------------------------------------------------------------
# LoraConfig.__post_init__ — target_modules normalization
# ---------------------------------------------------------------------------

class TestLoraConfigPostInit:
    def test_all_string_normalized_to_all_linear(self):
        lora = LoraConfig(target_modules="all")
        assert lora.target_modules == "all-linear"

    def test_all_in_list_normalized(self):
        lora = LoraConfig(target_modules=["all"])
        assert lora.target_modules == "all-linear"

    def test_all_linear_in_list_normalized_to_string(self):
        lora = LoraConfig(target_modules=["all-linear"])
        assert lora.target_modules == "all-linear"

    def test_all_linear_string_unchanged(self):
        lora = LoraConfig(target_modules="all-linear")
        assert lora.target_modules == "all-linear"

    def test_explicit_module_list_unchanged(self):
        modules = ["q_proj", "k_proj"]
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
