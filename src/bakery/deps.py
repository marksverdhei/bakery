"""Auto-install missing optional dependencies at runtime.

Checks model requirements after loading config and installs packages
that are needed but not present. Respects HF_HUB_OFFLINE and the
auto_install_optional_deps config flag.
"""

import importlib
import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)

# Maps model_type to optional packages needed for efficient training.
# Each entry is (import_name, pip_name, reason).
MODEL_OPTIONAL_DEPS: dict[str, list[tuple[str, str, str]]] = {
    "qwen3_5": [
        ("fla", "flash-linear-attention", "memory-efficient linear attention"),
    ],
}

# Packages needed for specific features.
FEATURE_OPTIONAL_DEPS: dict[str, list[tuple[str, str, str]]] = {
    "qlora": [
        ("bitsandbytes", "bitsandbytes", "4-bit quantization"),
    ],
}


def _is_offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "0") == "1"


def _is_installed(import_name: str) -> bool:
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False


def _install(pip_name: str) -> bool:
    """Install a package using pip. Returns True on success."""
    logger.info("Auto-installing %s...", pip_name)
    print(f"  Auto-installing {pip_name}...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pip_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.warning("Failed to install %s: %s", pip_name, e)
        print(f"  Warning: failed to install {pip_name}")
        return False


def ensure_deps(model_type: str | None = None, features: list[str] | None = None):
    """Install missing optional dependencies for the given model and features.

    Args:
        model_type: The model's model_type from its config (e.g. "qwen3_5").
        features: List of feature keys to check (e.g. ["qlora"]).
    """
    if _is_offline():
        return

    deps_to_check: list[tuple[str, str, str]] = []

    if model_type and model_type in MODEL_OPTIONAL_DEPS:
        deps_to_check.extend(MODEL_OPTIONAL_DEPS[model_type])

    for feature in features or []:
        if feature in FEATURE_OPTIONAL_DEPS:
            deps_to_check.extend(FEATURE_OPTIONAL_DEPS[feature])

    for import_name, pip_name, reason in deps_to_check:
        if not _is_installed(import_name):
            _install(pip_name)
