"""Bakery - Where LLMs go to get baked.

Prompt baking via KL divergence distillation with LoRA.
"""

from bakery.config import BakeryConfig, DataConfig, LoraConfig
from bakery.trainer import PromptBakingTrainer
from bakery.data import create_dataset, prompt_baking_collator, load_dataset
from bakery.kl import compute_kl_divergence

__all__ = [
    "BakeryConfig",
    "DataConfig",
    "LoraConfig",
    "PromptBakingTrainer",
    "create_dataset",
    "prompt_baking_collator",
    "load_dataset",
    "compute_kl_divergence",
]
