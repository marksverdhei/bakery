"""Tests for bakery.deps — auto-install optional dependencies."""

import importlib
from unittest.mock import MagicMock, patch, call

import pytest

from bakery.deps import (
    _is_offline,
    _is_installed,
    _install,
    ensure_deps,
)


# ---------------------------------------------------------------------------
# _is_offline
# ---------------------------------------------------------------------------

class TestIsOffline:
    def test_offline_when_env_set_to_one(self, monkeypatch):
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        assert _is_offline() is True

    def test_online_when_env_not_set(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        assert _is_offline() is False

    def test_online_when_env_set_to_zero(self, monkeypatch):
        monkeypatch.setenv("HF_HUB_OFFLINE", "0")
        assert _is_offline() is False

    def test_online_when_env_set_to_other_value(self, monkeypatch):
        monkeypatch.setenv("HF_HUB_OFFLINE", "true")
        assert _is_offline() is False


# ---------------------------------------------------------------------------
# _is_installed
# ---------------------------------------------------------------------------

class TestIsInstalled:
    def test_returns_true_for_installed_package(self):
        # json is always available
        assert _is_installed("json") is True

    def test_returns_false_for_missing_package(self):
        assert _is_installed("nonexistent_package_xyz_12345") is False

    def test_uses_importlib(self):
        with patch("importlib.import_module", return_value=MagicMock()) as mock_import:
            result = _is_installed("some_pkg")
        mock_import.assert_called_once_with("some_pkg")
        assert result is True

    def test_import_error_returns_false(self):
        with patch("importlib.import_module", side_effect=ImportError):
            result = _is_installed("some_pkg")
        assert result is False


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

    def test_calls_pip_install(self):
        import sys
        with patch("subprocess.check_call") as mock_call:
            _install("my-pkg")
        args = mock_call.call_args[0][0]
        assert sys.executable in args
        assert "-m" in args
        assert "pip" in args
        assert "install" in args
        assert "my-pkg" in args

    def test_prints_installing_message(self, capsys):
        with patch("subprocess.check_call", return_value=0):
            _install("some-package")
        out = capsys.readouterr().out
        assert "some-package" in out


# ---------------------------------------------------------------------------
# ensure_deps
# ---------------------------------------------------------------------------

class TestEnsureDeps:
    def test_does_nothing_when_offline(self, monkeypatch):
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        with patch("bakery.deps._install") as mock_install:
            ensure_deps(model_type="qwen3_5", features=["qlora"])
        mock_install.assert_not_called()

    def test_skips_already_installed_model_dep(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        with (
            patch("bakery.deps._is_installed", return_value=True),
            patch("bakery.deps._install") as mock_install,
        ):
            ensure_deps(model_type="qwen3_5")
        mock_install.assert_not_called()

    def test_installs_missing_model_dep(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        with (
            patch("bakery.deps._is_installed", return_value=False),
            patch("bakery.deps._install") as mock_install,
        ):
            ensure_deps(model_type="qwen3_5")
        mock_install.assert_called_once_with("flash-linear-attention")

    def test_installs_missing_feature_dep_qlora(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        with (
            patch("bakery.deps._is_installed", return_value=False),
            patch("bakery.deps._install") as mock_install,
        ):
            ensure_deps(features=["qlora"])
        mock_install.assert_called_once_with("bitsandbytes")

    def test_installs_missing_feature_dep_unsloth(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        with (
            patch("bakery.deps._is_installed", return_value=False),
            patch("bakery.deps._install") as mock_install,
        ):
            ensure_deps(features=["unsloth"])
        mock_install.assert_called_once_with("unsloth")

    def test_unknown_model_type_does_nothing(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        with patch("bakery.deps._install") as mock_install:
            ensure_deps(model_type="unknown_model_xyz")
        mock_install.assert_not_called()

    def test_unknown_feature_does_nothing(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        with patch("bakery.deps._install") as mock_install:
            ensure_deps(features=["nonexistent_feature"])
        mock_install.assert_not_called()

    def test_no_args_does_nothing(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        with patch("bakery.deps._install") as mock_install:
            ensure_deps()
        mock_install.assert_not_called()

    def test_multiple_features_all_installed_skips(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        with (
            patch("bakery.deps._is_installed", return_value=True),
            patch("bakery.deps._install") as mock_install,
        ):
            ensure_deps(features=["qlora", "unsloth"])
        mock_install.assert_not_called()

    def test_multiple_features_missing_installs_both(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        with (
            patch("bakery.deps._is_installed", return_value=False),
            patch("bakery.deps._install") as mock_install,
        ):
            ensure_deps(features=["qlora", "unsloth"])
        installed = {c[0][0] for c in mock_install.call_args_list}
        assert installed == {"bitsandbytes", "unsloth"}
