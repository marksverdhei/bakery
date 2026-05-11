"""Bakery - Where LLMs go to get baked.

Context baking (prefix-context distillation) via KL divergence with LoRA.
"""

from bakery.config import BakeryConfig, ContextConfig, DataConfig, LoraConfig
from bakery.trainer import ContextBakingTrainer, PromptBakingTrainer
from bakery.data import (
    create_conversational_dataset,
    create_dataset,
    load_conversations,
    load_dataset,
    prompt_baking_collator,
)
from bakery.kl import compute_kl_divergence
from bakery.masking import build_target_mask

__all__ = [
    "BakeryConfig",
    "ContextConfig",
    "DataConfig",
    "LoraConfig",
    "ContextBakingTrainer",
    "PromptBakingTrainer",
    "create_conversational_dataset",
    "create_dataset",
    "load_conversations",
    "load_dataset",
    "prompt_baking_collator",
    "compute_kl_divergence",
    "build_target_mask",
]
