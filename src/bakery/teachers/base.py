"""TeacherBackend abstract interface + TopKLogprobs sparse view."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class TopKLogprobs:
    """Sparse top-k view of a teacher's distribution over tokens.

    `indices[b, t, k]` is a vocab id; `values[b, t, k]` is its logprob under the
    teacher *renormalized over the top-k support* (so `exp(values).sum(-1) ≈ 1`).
    Renormalizing on the teacher side keeps the KL math identical between dense
    and sparse teachers — the student is scored only on those K positions.

    `attention_mask[b, t]` is 1 at positions the teacher actually scored.
    """

    indices: torch.Tensor  # (B, T, K) long
    values: torch.Tensor  # (B, T, K) float (log-probabilities, renormalized over K)
    attention_mask: torch.Tensor  # (B, T) long/bool

    @property
    def is_dense(self) -> bool:
        """True if K covers the full vocab (no truncation)."""
        return (
            self.indices.shape[-1] == self.values.shape[-1]
            and self.indices.shape[-1] >= 32000
        )  # heuristic; cheap enough not to matter


class TeacherBackend(ABC):
    """A model that scores token sequences and returns top-k logprobs.

    Backends MAY also implement `generate` for on-the-fly trajectory sampling,
    but the trainer's loss path only needs `score`.
    """

    @abstractmethod
    def score(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        top_k: int,
    ) -> TopKLogprobs:
        """Score a batch of student-tokenized sequences.

        Args:
            input_ids: (B, T) student-tokenizer ids.
            attention_mask: (B, T) 1=real, 0=padding.
            top_k: how many tokens to keep per position. Backends that can only
                provide ≤K (e.g. an API with `top_logprobs=20`) should return
                whatever they have and the trainer will reduce K to match.

        Returns:
            TopKLogprobs with the teacher's top-k at each position. Values are
            renormalized over those K so exp(values).sum(-1) ≈ 1.
        """

    def generate(self, messages: List[dict], max_new_tokens: int = 256) -> str:
        """Sample a completion conditioned on `messages`. Optional.

        Default: NotImplemented. Backends that support generation override this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement generate()"
        )

    @property
    def name(self) -> str:
        return type(self).__name__


def topk_from_logits(
    logits: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute top-k indices and renormalized logprobs from dense logits.

    Helper for backends that have full logits available (HF, vLLM with raw output).
    Returns `(indices, values)` where values are renormalized over the top-k.
    """
    k = min(top_k, logits.shape[-1])
    top_values, top_indices = torch.topk(logits, k=k, dim=-1)  # (..., K)
    # Convert raw logit values to logprobs renormalized over the top-k support.
    top_logprobs = top_values - top_values.logsumexp(dim=-1, keepdim=True)
    return top_indices, top_logprobs


def make_teacher(teacher_config) -> Optional[TeacherBackend]:
    """Build a TeacherBackend from a TeacherConfig, or None for local-toggle mode."""
    backend = (teacher_config.teacher_backend or "local-toggle").lower()
    if backend in ("local-toggle", "local", "self", "none", ""):
        return None
    if backend == "hf":
        from bakery.teachers.hf import HFTeacher

        return HFTeacher(
            model_name_or_path=teacher_config.teacher_model_name_or_path,
            torch_dtype=teacher_config.teacher_torch_dtype,
            device=teacher_config.teacher_device,
            trust_remote_code=teacher_config.teacher_trust_remote_code,
            attn_implementation=teacher_config.teacher_attn_implementation,
        )
    if backend == "vllm":
        from bakery.teachers.vllm import VLLMTeacher

        return VLLMTeacher(
            api_base=teacher_config.teacher_api_base,
            api_key=teacher_config.teacher_api_key,
            model=teacher_config.teacher_api_model
            or teacher_config.teacher_model_name_or_path,
        )
    if backend in ("openai", "openai-compat", "openai_compat"):
        from bakery.teachers.openai_compat import OpenAIAPITeacher

        return OpenAIAPITeacher(
            api_base=teacher_config.teacher_api_base,
            api_key=teacher_config.teacher_api_key,
            model=teacher_config.teacher_api_model
            or teacher_config.teacher_model_name_or_path,
        )
    raise ValueError(
        f"Unknown teacher backend: {backend!r}. "
        "Expected one of: local-toggle, hf, vllm, openai."
    )
