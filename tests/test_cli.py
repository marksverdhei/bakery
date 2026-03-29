"""Tests for bakery.cli — DTYPE_MAP and pre-parser."""

import sys
import argparse
from unittest.mock import patch, MagicMock

import torch
import pytest

from bakery.cli import DTYPE_MAP


class TestDtypeMap:
    def test_float32_mapped(self):
        assert DTYPE_MAP["float32"] is torch.float32

    def test_float16_mapped(self):
        assert DTYPE_MAP["float16"] is torch.float16

    def test_bfloat16_mapped(self):
        assert DTYPE_MAP["bfloat16"] is torch.bfloat16

    def test_unknown_returns_none(self):
        assert DTYPE_MAP.get("unknown_dtype") is None

    def test_all_values_are_torch_dtypes(self):
        for dtype in DTYPE_MAP.values():
            assert isinstance(dtype, torch.dtype)

    def test_map_has_three_entries(self):
        assert len(DTYPE_MAP) == 3


class TestPreParser:
    """Test the pre-parser built inside main()."""

    def _build_pre_parser(self):
        """Replicate the pre-parser construction from main()."""
        pre_parser = argparse.ArgumentParser(prog="bakery")
        pre_parser.add_argument("--config", required=True, metavar="FILE")
        return pre_parser

    def test_config_flag_required(self):
        pre_parser = self._build_pre_parser()
        with pytest.raises(SystemExit):
            pre_parser.parse_known_args([])

    def test_config_flag_accepted(self):
        pre_parser = self._build_pre_parser()
        args, _ = pre_parser.parse_known_args(["--config", "config.yaml"])
        assert args.config == "config.yaml"

    def test_unknown_args_passed_through(self):
        pre_parser = self._build_pre_parser()
        _, remaining = pre_parser.parse_known_args(
            ["--config", "cfg.yaml", "--learning_rate", "1e-4"]
        )
        assert "--learning_rate" in remaining
        assert "1e-4" in remaining

    def test_only_config_in_pre_parsed_namespace(self):
        pre_parser = self._build_pre_parser()
        args, _ = pre_parser.parse_known_args(["--config", "x.yaml"])
        assert hasattr(args, "config")
        assert not hasattr(args, "learning_rate")
