"""Bakery - Where LLMs go to get baked.

Prompt baking via KL divergence distillation with LoRA.
"""

from bakery.config import BakeryConfig, DataConfig, LoraConfig
from bakery.trainer import PromptBakingTrainer
from bakery.data import PromptBakingDataset, prompt_baking_collator, load_dataset_pairs
from bakery.kl import compute_kl_divergence

__all__ = [
    "BakeryConfig",
    "DataConfig",
    "LoraConfig",
    "PromptBakingTrainer",
    "PromptBakingDataset",
    "prompt_baking_collator",
    "load_dataset_pairs",
    "compute_kl_divergence",
]
