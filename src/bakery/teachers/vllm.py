"""vLLM teacher backend.

Two flavors share this file:

  1. vLLM-as-OpenAI-server: hit `/v1/completions` with `echo=True, max_tokens=0,
     logprobs=K` to score arbitrary student-tokenized sequences. This is the
     production path that mirrors TRL's pattern.

  2. vLLM-as-LLM (in-process): when the user prefers, instantiate `vllm.LLM`
     directly. Returns full dense logprobs without HTTP overhead.

For now the OpenAI-compat variant inherits from OpenAIAPITeacher and the
in-process variant is a thin wrapper. Full implementations land alongside the
TRL-style vLLM integration in a follow-up commit.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from bakery.teachers.base import TeacherBackend, TopKLogprobs
from bakery.teachers.openai_compat import OpenAIAPITeacher


class VLLMTeacher(OpenAIAPITeacher):
    """vLLM-as-OpenAI-server teacher.

    Inherits HTTP plumbing from OpenAIAPITeacher; assumes echo+logprobs are
    supported (vLLM's default).
    """

    def __init__(
        self,
        api_base: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
    ):
        super().__init__(
            api_base=api_base,
            api_key=api_key,
            model=model,
            echo_supported=True,
            timeout=timeout,
        )


class VLLMInProcessTeacher(TeacherBackend):
    """In-process vLLM teacher — to be filled in alongside TRL-style integration."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "VLLMInProcessTeacher: not yet implemented. Use VLLMTeacher (HTTP) or "
            "HFTeacher for now."
        )

    def score(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        top_k: int,
    ) -> TopKLogprobs:
        raise NotImplementedError

    def generate(self, messages: List[dict], max_new_tokens: int = 256) -> str:
        raise NotImplementedError
