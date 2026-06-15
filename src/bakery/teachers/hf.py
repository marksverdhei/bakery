"""HuggingFace in-process teacher backend.

Loads a separate Transformers model and scores student-tokenized sequences.
Same-tokenizer-as-student assumption — caller is responsible for that.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from bakery.teachers.base import TeacherBackend, TopKLogprobs, topk_from_logits

logger = logging.getLogger(__name__)


_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


class HFTeacher(TeacherBackend):
    """Score and generate using a HuggingFace Transformers model in-process.

    Designed to share GPU memory with the student where possible (the student is
    typically a smaller model). For larger teachers, consider running them on a
    separate device or via vLLM.
    """

    def __init__(
        self,
        model_name_or_path: str,
        torch_dtype: str = "bfloat16",
        device: Optional[str] = None,
        trust_remote_code: bool = False,
        attn_implementation: Optional[str] = None,
    ):
        if not model_name_or_path:
            raise ValueError("HFTeacher requires model_name_or_path")

        self.model_name_or_path = model_name_or_path
        dtype = _DTYPE_MAP.get(torch_dtype, torch.bfloat16)

        load_kwargs = dict(
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation
        if device:
            load_kwargs["device_map"] = device

        logger.info("Loading HF teacher: %s (dtype=%s)", model_name_or_path, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, **load_kwargs
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._gen_config = GenerationConfig(
            max_new_tokens=256,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    @torch.no_grad()
    def score(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        top_k: int,
    ) -> TopKLogprobs:
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        fwd = dict(input_ids=input_ids, attention_mask=attention_mask)
        mtype = getattr(self.model.config, "model_type", None)
        if mtype in ("gemma3", "gemma4", "gemma4_text"):
            fwd["token_type_ids"] = torch.zeros_like(input_ids)
            if mtype in ("gemma4", "gemma4_text"):
                fwd["mm_token_type_ids"] = torch.zeros_like(input_ids)

        logits = self.model(**fwd).logits  # (B, T, V)
        indices, values = topk_from_logits(logits, top_k)
        return TopKLogprobs(
            indices=indices,
            values=values,
            attention_mask=attention_mask,
        )

    @torch.no_grad()
    def generate(self, messages: List[dict], max_new_tokens: int = 256) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        gen_config = GenerationConfig(
            **{**self._gen_config.to_dict(), "max_new_tokens": max_new_tokens}
        )
        out = self.model.generate(**inputs, generation_config=gen_config)
        text = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return text.strip()
