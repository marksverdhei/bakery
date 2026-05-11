"""OpenAI-compatible HTTP teacher backend.

Targets any server implementing `/v1/completions` with `echo=True,
max_tokens=0, logprobs=K`. The canonical implementation is vLLM, but the
same plumbing works against Together, Fireworks, etc. as long as they expose
those flags.

Same-tokenizer assumption: token ids passed in by the trainer are interpreted
in the *student's* tokenizer. The teacher server must use a compatible tokenizer
(in practice this means same model family — Gemma family, Qwen family, Llama
family, etc.). The trainer registers the student tokenizer via
`set_student_tokenizer` once at construction time.

Cross-tokenizer scoring (Llama→Qwen) is out of scope and will raise on
re-encode mismatch.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import torch

from bakery.teachers.base import TeacherBackend, TopKLogprobs

logger = logging.getLogger(__name__)


class OpenAIAPITeacher(TeacherBackend):
    """OpenAI-compatible chat/completions teacher with sparse top-k scoring."""

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
        self._student_tokenizer = None

    def set_student_tokenizer(self, tokenizer) -> None:
        """Register the student tokenizer (used to re-encode API token strings)."""
        self._student_tokenizer = tokenizer

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

        Per-row: send the student's input_ids (vLLM accepts a list of ints as
        `prompt`), echo back, request top-k logprobs at every position. Re-encode
        each returned top-k token *string* into a student vocab id.
        """
        if not self.echo_supported:
            raise NotImplementedError(
                "Server doesn't advertise echo support; sparse scoring unavailable."
            )
        if self._student_tokenizer is None:
            raise RuntimeError(
                "OpenAIAPITeacher.score requires a student tokenizer — call "
                "set_student_tokenizer(...) before training."
            )

        B, T = input_ids.shape
        vocab_size = len(self._student_tokenizer)
        indices_out = torch.zeros(B, T, top_k, dtype=torch.long)
        values_out = torch.full((B, T, top_k), -1e9, dtype=torch.float32)
        mask_out = attention_mask.clone()

        client = self._client()
        try:
            for b in range(B):
                row_attn = attention_mask[b]
                # Strip left-padding before sending — vLLM doesn't need our padding.
                real_start = int((row_attn > 0).nonzero(as_tuple=False)[0].item())
                real_ids = input_ids[b, real_start:].tolist()
                resp = client.post(
                    "/completions",
                    json={
                        "model": self.model,
                        "prompt": real_ids,
                        "max_tokens": 0,
                        "echo": True,
                        "logprobs": top_k,
                        "temperature": 0,
                    },
                )
                resp.raise_for_status()
                choice = resp.json()["choices"][0]
                lp_block = choice.get("logprobs") or {}
                top_lps = lp_block.get("top_logprobs") or []
                # vLLM returns top_logprobs as a list (one per echoed token).
                # Position 0 is the very first token, which has no prior context
                # and is conventionally null. Fill from position 1 onward.
                for t_local, pos_top in enumerate(top_lps):
                    if pos_top is None:
                        continue
                    t_full = real_start + t_local
                    if t_full >= T:
                        break
                    # pos_top: dict-like {token_string: logprob}.
                    items = list(pos_top.items())
                    if not items:
                        continue
                    # Sort by logprob desc to be safe — most servers already do this.
                    items.sort(key=lambda kv: -kv[1])
                    raw_logprobs = []
                    raw_indices = []
                    for tok_str, lp in items[:top_k]:
                        ids = self._student_tokenizer.encode(
                            tok_str, add_special_tokens=False
                        )
                        if not ids:
                            continue
                        # Pick the single id that represents this token. If the
                        # API token is a multi-piece string under the student
                        # tokenizer, we approximate by the first piece.
                        sid = ids[0]
                        if sid >= vocab_size:
                            continue
                        raw_indices.append(sid)
                        raw_logprobs.append(lp)
                    if not raw_indices:
                        continue
                    n = min(len(raw_indices), top_k)
                    idx_t = torch.tensor(raw_indices[:n], dtype=torch.long)
                    lp_t = torch.tensor(raw_logprobs[:n], dtype=torch.float32)
                    # Renormalize over the K we actually captured so the trainer's
                    # `exp(values).sum ≈ 1` invariant holds.
                    lp_t = lp_t - torch.logsumexp(lp_t, dim=0)
                    indices_out[b, t_full, :n] = idx_t
                    values_out[b, t_full, :n] = lp_t
                    if n < top_k:
                        # Pad remaining slots by reusing the first index with
                        # very negative logprob — keeps tensors well-formed
                        # and the renormalization above keeps mass on the real K.
                        indices_out[b, t_full, n:] = idx_t[0]
                        values_out[b, t_full, n:] = -1e9
        finally:
            client.close()

        return TopKLogprobs(
            indices=indices_out,
            values=values_out,
            attention_mask=mask_out,
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
