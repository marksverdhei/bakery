"""Tests for bakery.deps — auto-install optional dependencies."""

from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import pytest

from bakery.deps import (
    _is_offline,
    _is_installed,
    _install,
    ensure_deps,
    MODEL_OPTIONAL_DEPS,
    FEATURE_OPTIONAL_DEPS,
)


# ---------------------------------------------------------------------------
# _is_offline
# ---------------------------------------------------------------------------

class TestIsOffline:
    def test_not_offline_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HF_HUB_OFFLINE", None)
            assert _is_offline() is False

    def test_offline_when_env_is_1(self):
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}):
            assert _is_offline() is True

    def test_not_offline_when_env_is_0(self):
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "0"}):
            assert _is_offline() is False

    def test_not_offline_when_env_is_empty(self):
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": ""}):
            assert _is_offline() is False


# ---------------------------------------------------------------------------
# _is_installed
# ---------------------------------------------------------------------------

class TestIsInstalled:
    def test_known_module_is_installed(self):
        # `os` is always importable
        assert _is_installed("os") is True

    def test_missing_module_is_not_installed(self):
        assert _is_installed("__this_module_does_not_exist_42__") is False

    def test_top_level_package(self):
        # pytest is installed (we're running with it)
        assert _is_installed("pytest") is True


# ---------------------------------------------------------------------------
# _install
# ---------------------------------------------------------------------------

class TestInstall:
    def test_returns_true_on_success(self):
        with patch("subprocess.check_call", return_value=0):
            result = _install("some-package")
        assert result is True

    def test_returns_false_on_failure(self):
        import subprocess
        with patch("subprocess.check_call", side_effect=subprocess.CalledProcessError(1, "pip")):
            result = _install("some-package")
        assert result is False

    def test_calls_pip_with_package_name(self):
        with patch("subprocess.check_call") as mock_call:
            _install("my-package")
        call_args = mock_call.call_args[0][0]
        assert "my-package" in call_args
        assert "-m" in call_args
        assert "pip" in call_args


# ---------------------------------------------------------------------------
# ensure_deps
# ---------------------------------------------------------------------------

class TestEnsureDeps:
    def test_does_nothing_when_offline(self):
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}):
            with patch("bakery.deps._install") as mock_install:
                ensure_deps(model_type="qwen3_5", features=["qlora"])
        mock_install.assert_not_called()

    def test_does_nothing_when_no_model_type_or_features(self):
        with patch("bakery.deps._is_installed", return_value=True):
            with patch("bakery.deps._install") as mock_install:
                ensure_deps(model_type=None, features=None)
        mock_install.assert_not_called()

    def test_does_nothing_when_already_installed(self):
        with patch("bakery.deps._is_installed", return_value=True):
            with patch("bakery.deps._install") as mock_install:
                ensure_deps(model_type="qwen3_5", features=["qlora"])
        mock_install.assert_not_called()

    def test_installs_missing_model_dep(self):
        with patch("bakery.deps._is_installed", return_value=False):
            with patch("bakery.deps._install") as mock_install:
                ensure_deps(model_type="qwen3_5", features=[])
        # qwen3_5 requires flash-linear-attention
        installed_packages = [call[0][0] for call in mock_install.call_args_list]
        assert "flash-linear-attention" in installed_packages

    def test_installs_missing_qlora_dep(self):
        with patch("bakery.deps._is_installed", return_value=False):
            with patch("bakery.deps._install") as mock_install:
                ensure_deps(model_type=None, features=["qlora"])
        installed_packages = [call[0][0] for call in mock_install.call_args_list]
        assert "bitsandbytes" in installed_packages

    def test_installs_missing_unsloth_dep(self):
        with patch("bakery.deps._is_installed", return_value=False):
            with patch("bakery.deps._install") as mock_install:
                ensure_deps(model_type=None, features=["unsloth"])
        installed_packages = [call[0][0] for call in mock_install.call_args_list]
        assert "unsloth" in installed_packages

    def test_unknown_model_type_is_ignored(self):
        with patch("bakery.deps._install") as mock_install:
            ensure_deps(model_type="unknown_model_xyz", features=[])
        mock_install.assert_not_called()

    def test_unknown_feature_is_ignored(self):
        with patch("bakery.deps._install") as mock_install:
            ensure_deps(model_type=None, features=["not_a_real_feature"])
        mock_install.assert_not_called()

    def test_multiple_features_all_installed(self):
        with patch("bakery.deps._is_installed", return_value=False):
            with patch("bakery.deps._install") as mock_install:
                ensure_deps(model_type=None, features=["qlora", "unsloth"])
        installed_packages = {call[0][0] for call in mock_install.call_args_list}
        assert "bitsandbytes" in installed_packages
        assert "unsloth" in installed_packages

    def test_no_install_for_empty_features_list(self):
        with patch("bakery.deps._install") as mock_install:
            ensure_deps(model_type=None, features=[])
        mock_install.assert_not_called()


# ---------------------------------------------------------------------------
# Dependency tables
# ---------------------------------------------------------------------------

class TestDepsTables:
    def test_model_optional_deps_is_dict(self):
        assert isinstance(MODEL_OPTIONAL_DEPS, dict)

    def test_feature_optional_deps_is_dict(self):
        assert isinstance(FEATURE_OPTIONAL_DEPS, dict)

    def test_qlora_in_feature_deps(self):
        assert "qlora" in FEATURE_OPTIONAL_DEPS

    def test_unsloth_in_feature_deps(self):
        assert "unsloth" in FEATURE_OPTIONAL_DEPS

    def test_all_entries_are_triples(self):
        for key, entries in {**MODEL_OPTIONAL_DEPS, **FEATURE_OPTIONAL_DEPS}.items():
            for entry in entries:
                assert len(entry) == 3, f"{key}: entry {entry!r} is not a 3-tuple"
