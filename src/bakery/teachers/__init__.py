"""Teacher backends for GKD / context distillation.

The trainer asks a `TeacherBackend` to score token sequences. Three backends:

- `HFTeacher`        — load a separate HuggingFace model in-process (fallback / dev path).
- `VLLMTeacher`      — query a vLLM server via its OpenAI-compatible HTTP API (production).
- `OpenAIAPITeacher` — query any OpenAI-compatible endpoint that exposes `top_logprobs`.

All backends return the same shape: `TopKLogprobs`, a sparse top-k view of the teacher's
distribution at each scored position. The trainer's KL loss consumes that uniformly,
regardless of whether K is the full vocab (HF dense) or K=20 (API sparse).

A *student-side same-tokenizer assumption* applies: token ids passed to a teacher backend
are interpreted in the student's tokenizer. Mixing tokenizers (e.g. Llama→Qwen) is out of
scope for now.
"""

from bakery.teachers.base import TeacherBackend, TopKLogprobs, make_teacher
from bakery.teachers.hf import HFTeacher

__all__ = ["TeacherBackend", "TopKLogprobs", "HFTeacher", "make_teacher"]
