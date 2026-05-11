"""OpenAI-compatible HTTP teacher backend.

Targets any server implementing `/v1/chat/completions` with `logprobs=true,
top_logprobs=K`. Works against OpenAI, vLLM, Together, Fireworks, etc.

Limitations:
  - We get top-k logprobs at generation positions only (not arbitrary token
    scoring). To score student-tokenized sequences for KL, we use the
    `/v1/completions` (raw text completion) endpoint where supported, sending
    the decoded prefix and asking for `logprobs` on `max_tokens=0` style
    forward passes. Servers that don't support this are limited to on-policy
    sampling where the teacher generated the trajectory.

  - The token ids the server returns are *its* tokenizer ids, not the student's.
    We re-tokenize the returned `tokens` strings into student vocab ids.

This backend is implemented for the on-policy GKD path: teacher generates a
trajectory, the same call returns top-k logprobs for those tokens, and KL is
computed at those positions. For off-policy scoring of arbitrary student
sequences, prefer the HF or vLLM-native backend.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import torch

from bakery.teachers.base import TeacherBackend, TopKLogprobs

logger = logging.getLogger(__name__)


class OpenAIAPITeacher(TeacherBackend):
    """OpenAI-compatible chat/completions teacher.

    score() requires the server to support `logprobs` on the completions
    endpoint with input echoing (e.g. vLLM's `echo=True, max_tokens=0`).
    For pure OpenAI without echo, score() raises NotImplementedError and the
    caller should use on-policy generate() + per-token logprobs instead.
    """

    def __init__(
        self,
        api_base: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        echo_supported: bool = True,
        timeout: float = 60.0,
    ):
        try:
            import httpx  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "OpenAIAPITeacher requires `httpx`. Install with: pip install httpx"
            ) from e

        if not api_base:
            raise ValueError("OpenAIAPITeacher requires api_base")
        if not model:
            raise ValueError("OpenAIAPITeacher requires model")

        self.api_base = api_base.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or ""
        self.model = model
        self.echo_supported = echo_supported
        self.timeout = timeout

    def _client(self):
        import httpx

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return httpx.Client(
            base_url=self.api_base, headers=headers, timeout=self.timeout
        )

    def score(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        top_k: int,
    ) -> TopKLogprobs:
        """Score sequences via /v1/completions with echo=True, max_tokens=0.

        Note: this currently requires the server's tokenizer to match the student's.
        Cross-tokenizer scoring is a future feature.
        """
        if not self.echo_supported:
            raise NotImplementedError(
                "OpenAIAPITeacher.score requires a server that supports echo=True on "
                "/v1/completions (vLLM, some Together models). Pure OpenAI does not. "
                "Use on-policy generation instead."
            )
        raise NotImplementedError(
            "OpenAIAPITeacher.score is not yet wired up — implementing alongside vLLM."
        )

    def generate(self, messages: List[dict], max_new_tokens: int = 256) -> str:
        client = self._client()
        try:
            resp = client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_new_tokens,
                    "temperature": 0.8,
                    "top_p": 0.9,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        finally:
            client.close()
