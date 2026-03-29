"""Tests for bakery CLI argument parsing and YAML config handling."""

from __future__ import annotations

import argparse
import os
import tempfile
import textwrap

import pytest
import yaml

from transformers import HfArgumentParser

from bakery.config import BakeryConfig, DataConfig, LoraConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(content: dict, path: str) -> None:
    with open(path, "w") as f:
        yaml.dump(content, f)


def _minimal_yaml(tmp_path, extra: dict | None = None) -> str:
    """Write the minimum YAML that HfArgumentParser can parse."""
    base = {
        "output_dir": str(tmp_path / "out"),
        "system_prompt": "You are helpful.",
        "model_name_or_path": "gpt2",
        "num_train_epochs": 1,
    }
    if extra:
        base.update(extra)
    path = str(tmp_path / "config.yaml")
    _write_yaml(base, path)
    return path


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

class TestYamlLoading:
    def test_minimal_yaml_parsed(self, tmp_path):
        path = _minimal_yaml(tmp_path)
        parser = HfArgumentParser((BakeryConfig, DataConfig, LoraConfig))
        baking, data, lora = parser.parse_yaml_file(path, allow_extra_keys=True)
        assert baking.system_prompt == "You are helpful."
        assert data.model_name_or_path == "gpt2"

    def test_output_dir_from_yaml(self, tmp_path):
        out_dir = str(tmp_path / "myoutput")
        path = _minimal_yaml(tmp_path, {"output_dir": out_dir})
        parser = HfArgumentParser((BakeryConfig, DataConfig, LoraConfig))
        baking, _, _ = parser.parse_yaml_file(path, allow_extra_keys=True)
        assert baking.output_dir == out_dir

    def test_lora_config_from_yaml(self, tmp_path):
        path = _minimal_yaml(tmp_path, {"r": 16, "lora_alpha": 32})
        parser = HfArgumentParser((BakeryConfig, DataConfig, LoraConfig))
        _, _, lora = parser.parse_yaml_file(path, allow_extra_keys=True)
        assert lora.r == 16
        assert lora.lora_alpha == 32

    def test_numeric_floats_as_strings(self, tmp_path):
        """HfArgumentParser from YAML may deliver learning_rate as string."""
        path = _minimal_yaml(tmp_path, {"learning_rate": "3e-4"})
        parser = HfArgumentParser((BakeryConfig, DataConfig, LoraConfig))
        # BakeryConfig.__post_init__ should coerce string → float
        baking, _, _ = parser.parse_yaml_file(path, allow_extra_keys=True)
        assert abs(baking.learning_rate - 3e-4) < 1e-10

    def test_target_modules_all_normalized(self, tmp_path):
        path = _minimal_yaml(tmp_path, {"target_modules": "all"})
        parser = HfArgumentParser((BakeryConfig, DataConfig, LoraConfig))
        _, _, lora = parser.parse_yaml_file(path, allow_extra_keys=True)
        assert lora.target_modules == "all-linear"

    def test_target_modules_list_all_normalized(self, tmp_path):
        path = _minimal_yaml(tmp_path, {"target_modules": ["all"]})
        parser = HfArgumentParser((BakeryConfig, DataConfig, LoraConfig))
        _, _, lora = parser.parse_yaml_file(path, allow_extra_keys=True)
        assert lora.target_modules == "all-linear"

    def test_extra_keys_allowed(self, tmp_path):
        """allow_extra_keys=True should not raise on unknown fields."""
        path = _minimal_yaml(tmp_path, {"unknown_field_xyz": "ignored"})
        parser = HfArgumentParser((BakeryConfig, DataConfig, LoraConfig))
        # Should not raise
        baking, data, lora = parser.parse_yaml_file(path, allow_extra_keys=True)
        assert baking.system_prompt == "You are helpful."


# ---------------------------------------------------------------------------
# Pre-parser (--config extraction)
# ---------------------------------------------------------------------------

class TestPreParser:
    """Test the pre_parser that extracts --config before HfArgumentParser."""

    def _make_pre_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(prog="bakery")
        parser.add_argument("--config", required=True, metavar="FILE")
        return parser

    def test_extracts_config_path(self, tmp_path):
        parser = self._make_pre_parser()
        pre_args, remaining = parser.parse_known_args(["--config", "my.yaml"])
        assert pre_args.config == "my.yaml"

    def test_remaining_args_after_config(self, tmp_path):
        parser = self._make_pre_parser()
        _, remaining = parser.parse_known_args(
            ["--config", "my.yaml", "--learning_rate", "1e-4", "--num_train_epochs", "5"]
        )
        assert "--learning_rate" in remaining
        assert "1e-4" in remaining
        assert "--num_train_epochs" in remaining

    def test_config_required(self, tmp_path):
        parser = self._make_pre_parser()
        with pytest.raises(SystemExit):
            parser.parse_known_args(["--learning_rate", "1e-4"])

    def test_config_not_in_remaining(self, tmp_path):
        parser = self._make_pre_parser()
        _, remaining = parser.parse_known_args(["--config", "my.yaml", "--seed", "42"])
        assert "--config" not in remaining
        assert "my.yaml" not in remaining


# ---------------------------------------------------------------------------
# CLI override merging
# ---------------------------------------------------------------------------

class TestCliOverrideMerging:
    """Test that CLI args override YAML config fields correctly."""

    def _parse_with_overrides(self, tmp_path, yaml_extra: dict, cli_args: list):
        """Simulate the CLI's YAML + CLI override merging."""
        path = _minimal_yaml(tmp_path, yaml_extra)
        parser = HfArgumentParser((BakeryConfig, DataConfig, LoraConfig))
        baking, data, lora = parser.parse_yaml_file(path, allow_extra_keys=True)

        if not cli_args:
            return baking, data, lora

        explicit_keys = set()
        for arg in cli_args:
            if arg.startswith("--"):
                explicit_keys.add(arg.lstrip("-").replace("-", "_"))

        override_parser = HfArgumentParser((BakeryConfig, DataConfig, LoraConfig))
        overrides = override_parser.parse_args_into_dataclasses(
            args=["--output_dir", baking.output_dir] + cli_args,
            return_remaining_strings=True,
        )
        for override_cfg, base_cfg in zip(
            overrides[:3],
            (baking, data, lora),
        ):
            for k, v in vars(override_cfg).items():
                if k in explicit_keys:
                    setattr(base_cfg, k, v)

        return baking, data, lora

    def test_cli_overrides_yaml_learning_rate(self, tmp_path):
        baking, _, _ = self._parse_with_overrides(
            tmp_path,
            yaml_extra={"learning_rate": "1e-4"},
            cli_args=["--learning_rate", "5e-5"],
        )
        assert abs(baking.learning_rate - 5e-5) < 1e-10

    def test_yaml_value_kept_when_no_cli_override(self, tmp_path):
        baking, _, _ = self._parse_with_overrides(
            tmp_path,
            yaml_extra={"num_train_epochs": 7},
            cli_args=[],
        )
        assert baking.num_train_epochs == 7

    def test_cli_overrides_yaml_seed(self, tmp_path):
        baking, _, _ = self._parse_with_overrides(
            tmp_path,
            yaml_extra={"seed": 42},
            cli_args=["--seed", "123"],
        )
        assert baking.seed == 123

    def test_multiple_cli_overrides(self, tmp_path):
        baking, _, _ = self._parse_with_overrides(
            tmp_path,
            yaml_extra={"num_train_epochs": 3, "seed": 1},
            cli_args=["--num_train_epochs", "10", "--seed", "99"],
        )
        assert baking.num_train_epochs == 10
        assert baking.seed == 99
