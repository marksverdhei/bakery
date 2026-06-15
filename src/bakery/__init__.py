"""Bakery - Where LLMs go to get baked.

Context baking (prefix-context distillation) via KL divergence with LoRA.
"""

from bakery.config import (
    BakeryConfig,
    ContextConfig,
    DataConfig,
    LoraConfig,
    TeacherConfig,
)
from bakery.trainer import ContextBakingTrainer, PromptBakingTrainer
from bakery.data import (
    create_conversational_dataset,
    create_dataset,
    load_conversations,
    load_dataset,
    prompt_baking_collator,
)
from bakery.kl import compute_kl_divergence, topk_forward_kl
from bakery.masking import build_target_mask
from bakery.teachers import HFTeacher, TeacherBackend, TopKLogprobs, make_teacher

__all__ = [
    "BakeryConfig",
    "ContextConfig",
    "DataConfig",
    "LoraConfig",
    "TeacherConfig",
    "ContextBakingTrainer",
    "PromptBakingTrainer",
    "create_conversational_dataset",
    "create_dataset",
    "load_conversations",
    "load_dataset",
    "prompt_baking_collator",
    "compute_kl_divergence",
    "topk_forward_kl",
    "build_target_mask",
    "HFTeacher",
    "TeacherBackend",
    "TopKLogprobs",
    "make_teacher",
]
