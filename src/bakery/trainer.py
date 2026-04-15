"""Context baking trainer via KL divergence.

Generalizes the original prompt-baking approach to arbitrary prefix contexts —
system prompts, conversation histories, accumulated memories, few-shot examples.
The teacher sees the full prefix; the student sees a trimmed version (or none).
KL divergence distills the teacher into the student, baking the prefix into weights.

Inherits from transformers.Trainer for standard HF infrastructure (logging,
checkpointing, schedulers, gradient accumulation).

Based on the Prompt Baking paper (arxiv 2409.13697), generalized.
"""

from __future__ import annotations

import logging
import warnings
from typing import Callable, List, Optional

import torch
from datasets import Dataset
from transformers import (
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
)
from transformers.trainer_utils import EvalPrediction

from bakery.kl import compute_kl_divergence, disable_adapters, padding_side
from bakery.masking import build_target_mask

logger = logging.getLogger(__name__)


class ContextBakingTrainer(Trainer):
    """Trainer that bakes an arbitrary prefix context into model weights via KL divergence.

    For each example:
      - Teacher view: prefix_messages + turns (+ response)
      - Student view: prefix_messages[-student_retained_turns:] + turns (+ response)
    KL is computed on tokens belonging to messages whose role matches
    `context_config.target_roles` and content matches `target_content_pattern`
    (if set). Prefix tokens never receive loss.
    """

    def __init__(
        self,
        model: PreTrainedModel | str | None = None,
        args=None,
        context_config=None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        processing_class: PreTrainedTokenizerBase | None = None,
        data_collator: Optional[Callable] = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
    ):
        self.num_trajectories = args.num_trajectories
        self.trajectory_length = args.trajectory_length
        self.sampling_temperature = args.sampling_temperature
        self.kl_temperature = args.temperature

        # Back-compat: if no ContextConfig is passed but args has system_prompt,
        # auto-desugar into a minimal ContextConfig. Keeps direct-instantiation
        # users on the old API working.
        if context_config is None and getattr(args, "system_prompt", None):
            from bakery.config import ContextConfig

            context_config = ContextConfig(
                prefix_messages=[
                    {"role": "system", "content": args.system_prompt}
                ]
            )

        # Context configuration — prefix, student view, target mask.
        self.context_config = context_config
        self.prefix_messages: List[dict] = (
            list(context_config.prefix_messages)
            if context_config and context_config.prefix_messages
            else []
        )
        self.student_retained_turns: int = (
            context_config.student_retained_turns if context_config else 0
        )
        self.target_roles: List[str] = (
            list(context_config.target_roles)
            if context_config and context_config.target_roles
            else ["assistant"]
        )
        self.target_content_pattern: Optional[str] = (
            context_config.target_content_pattern if context_config else None
        )

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self.model_accepts_loss_kwargs = False
        self.generation_config = GenerationConfig(
            max_new_tokens=self.trajectory_length,
            temperature=self.sampling_temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=self.processing_class.pad_token_id,
        )

    # -- Tokenization --

    def _tokenize(self, text, **kwargs) -> dict:
        max_len = getattr(self.args, "max_seq_length", None)
        if max_len:
            kwargs.setdefault("truncation", True)
            kwargs.setdefault("max_length", max_len)
        return self.processing_class(text, add_special_tokens=False, **kwargs)

    # -- Message assembly --

    def _row_prefix(self, row_prefix: Optional[List[dict]]) -> List[dict]:
        """Return the effective prefix for a row: per-row if set, else global."""
        if row_prefix:
            return list(row_prefix)
        return list(self.prefix_messages)

    def _student_prefix(self, prefix: List[dict]) -> List[dict]:
        """Trim the prefix down to the last `student_retained_turns` messages."""
        if self.student_retained_turns <= 0:
            return []
        return prefix[-self.student_retained_turns :]

    @staticmethod
    def _append_response(messages: List[dict], response: Optional[str]) -> List[dict]:
        if response:
            return list(messages) + [{"role": "assistant", "content": response}]
        return list(messages)

    def _build_example(
        self,
        row_prefix: Optional[List[dict]],
        turns: List[dict],
        response: Optional[str],
    ):
        """Build tokenized teacher + student views for a single example.

        Returns a dict with:
          teacher_ids, teacher_mask (target mask over teacher tokens),
          student_ids, student_mask (target mask over student tokens),
          teacher_prefix_token_len, student_prefix_token_len
        or None if no target tokens were found.
        """
        prefix = self._row_prefix(row_prefix)
        student_prefix = self._student_prefix(prefix)

        teacher_messages = self._append_response(list(prefix) + list(turns), response)
        student_messages = self._append_response(
            list(student_prefix) + list(turns), response
        )

        # Target mask: prefix messages never count as targets.
        teacher_ids, teacher_target_mask, teacher_first = build_target_mask(
            self.processing_class,
            teacher_messages,
            self.target_roles,
            self.target_content_pattern,
            target_min_msg_idx=len(prefix),
        )
        student_ids, student_target_mask, student_first = build_target_mask(
            self.processing_class,
            student_messages,
            self.target_roles,
            self.target_content_pattern,
            target_min_msg_idx=len(student_prefix),
        )

        if teacher_first >= len(teacher_ids) or student_first >= len(student_ids):
            return None  # no target tokens

        return {
            "teacher_ids": teacher_ids,
            "teacher_mask": teacher_target_mask,
            "teacher_first": teacher_first,
            "student_ids": student_ids,
            "student_mask": student_target_mask,
            "student_first": student_first,
        }

    # -- Batch preparation --

    @staticmethod
    def _normalize_batch(inputs: dict) -> tuple[list, list, list]:
        """Normalize batch into (prefix_messages_per_row, turns_per_row, responses).

        Accepts new-format batches (prefix_messages, turns, responses) and the
        legacy format (user_messages, responses). Legacy format wraps each user
        message as a single-turn [{role: user, content: msg}] list.
        """
        if "turns" in inputs:
            prefix_list = inputs.get("prefix_messages") or [None] * len(inputs["turns"])
            return (
                list(prefix_list),
                [list(t) for t in inputs["turns"]],
                list(inputs.get("responses", [None] * len(inputs["turns"]))),
            )
        # Legacy shape
        user_messages = inputs.get("user_messages", [])
        responses = inputs.get("responses", [None] * len(user_messages))
        turns = [[{"role": "user", "content": m}] for m in user_messages]
        prefix_list = [None] * len(user_messages)
        return prefix_list, turns, list(responses)

    def _build_batch(self, inputs, model) -> Optional[dict]:
        """Tokenize + pad the batch; return dict of tensors or None if empty.

        Returned dict:
          teacher_fwd: kwargs for teacher forward (input_ids, attention_mask, ...)
          student_fwd: kwargs for student forward
          teacher_mask_padded: (B, T_t) bool mask of target tokens (aligned with padding)
          student_mask_padded: (B, T_s) bool mask of target tokens
        """
        prefix_list, turns_list, responses = self._normalize_batch(inputs)
        if not turns_list:
            return None

        built = []
        for prefix, turns, resp in zip(prefix_list, turns_list, responses):
            # Require a response or a turn ending in assistant for KL to have a target.
            if not resp and not any(m.get("role") == "assistant" for m in turns):
                continue
            # If response is empty string, skip.
            if resp is not None and isinstance(resp, str) and not resp.strip():
                continue
            example = self._build_example(prefix, turns, resp)
            if example is None:
                continue
            built.append(example)

        if not built:
            return None

        # Left-pad teacher and student id lists into tensors.
        pad_id = self.processing_class.pad_token_id
        if pad_id is None:
            pad_id = self.processing_class.eos_token_id

        def _pad_left(seqs, masks):
            max_len = max(len(s) for s in seqs)
            out_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
            out_attn = torch.zeros((len(seqs), max_len), dtype=torch.long)
            out_mask = torch.zeros((len(seqs), max_len), dtype=torch.bool)
            for i, (ids, msk) in enumerate(zip(seqs, masks)):
                n = len(ids)
                out_ids[i, max_len - n :] = torch.tensor(ids, dtype=torch.long)
                out_attn[i, max_len - n :] = 1
                out_mask[i, max_len - n :] = msk
            return out_ids, out_attn, out_mask

        t_ids, t_attn, t_mask = _pad_left(
            [b["teacher_ids"] for b in built],
            [b["teacher_mask"] for b in built],
        )
        s_ids, s_attn, s_mask = _pad_left(
            [b["student_ids"] for b in built],
            [b["student_mask"] for b in built],
        )

        device = model.device
        t_ids = t_ids.to(device)
        t_attn = t_attn.to(device)
        s_ids = s_ids.to(device)
        s_attn = s_attn.to(device)
        # Keep masks on CPU until slicing; move to device per-sample in loss.

        teacher_fwd = {"input_ids": t_ids, "attention_mask": t_attn}
        student_fwd = {"input_ids": s_ids, "attention_mask": s_attn}
        self._inject_gemma_token_types(model, teacher_fwd, t_ids)
        self._inject_gemma_token_types(model, student_fwd, s_ids)

        return {
            "teacher_fwd": teacher_fwd,
            "student_fwd": student_fwd,
            "teacher_mask_padded": t_mask,
            "student_mask_padded": s_mask,
            "teacher_attn": t_attn,
            "student_attn": s_attn,
        }

    @staticmethod
    def _inject_gemma_token_types(model, fwd: dict, input_ids: torch.Tensor) -> None:
        """Gemma 3/4 require token_type_ids (and mm_token_type_ids for Gemma 4)
        during training even for text-only inputs. All-zeros is correct."""
        mtype = getattr(model.config, "model_type", None)
        if mtype in ("gemma3", "gemma4", "gemma4_text"):
            fwd["token_type_ids"] = torch.zeros_like(input_ids)
            if mtype in ("gemma4", "gemma4_text"):
                fwd["mm_token_type_ids"] = torch.zeros_like(input_ids)

    # -- KL loss from aligned logits + masks --

    def _kl_from_logits(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        teacher_mask_padded: torch.BoolTensor,
        student_mask_padded: torch.BoolTensor,
    ) -> Optional[torch.Tensor]:
        """Compute mean per-example KL over aligned target tokens.

        Logits at position t predict token t+1, so we shift by 1. Teacher and
        student may have different prefix lengths, so we align on the trainable
        region by finding the first target token in each and taking the common
        length.
        """
        losses = []
        B = teacher_logits.shape[0]
        device = student_logits.device

        for i in range(B):
            t_mask = teacher_mask_padded[i].to(device)
            s_mask = student_mask_padded[i].to(device)

            t_target_positions = t_mask.nonzero(as_tuple=False).squeeze(-1)
            s_target_positions = s_mask.nonzero(as_tuple=False).squeeze(-1)
            if t_target_positions.numel() == 0 or s_target_positions.numel() == 0:
                continue

            t_start = int(t_target_positions[0].item())
            s_start = int(s_target_positions[0].item())

            # Logits predict next token → shift by 1.
            t_logits = teacher_logits[i : i + 1, t_start - 1 : -1, :]
            s_logits = student_logits[i : i + 1, s_start - 1 : -1, :]
            t_tail_mask = t_mask[t_start:].float().unsqueeze(0)
            s_tail_mask = s_mask[s_start:].float().unsqueeze(0)

            min_len = min(
                t_logits.shape[1], s_logits.shape[1], t_tail_mask.shape[1], s_tail_mask.shape[1]
            )
            if min_len == 0:
                continue

            t_logits = t_logits[:, :min_len, :]
            s_logits = s_logits[:, :min_len, :]
            combined_mask = (t_tail_mask[:, :min_len] * s_tail_mask[:, :min_len])
            if combined_mask.sum() == 0:
                continue

            loss = compute_kl_divergence(
                t_logits.detach() if t_logits.requires_grad else t_logits,
                s_logits,
                combined_mask,
                self.kl_temperature,
            )
            losses.append(loss)

        if not losses:
            return None
        return torch.stack(losses).mean()

    # -- Trajectory generation --

    def _generate_trajectory(self, user_message: str) -> str:
        """Generate a response from the teacher (adapters disabled, full prefix visible)."""
        teacher_messages = list(self.prefix_messages) + [
            {"role": "user", "content": user_message}
        ]
        prompt = self.processing_class.apply_chat_template(
            teacher_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenize(prompt, return_tensors="pt").to(self.model.device)

        was_training = self.model.training
        self.model.eval()

        with torch.no_grad():
            with disable_adapters(self.model):
                outputs = self.model.generate(
                    **inputs, generation_config=self.generation_config
                )

        if was_training:
            self.model.train()

        response = self.processing_class.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return response.strip()

    # -- Loss + eval --

    def _zero_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.args.device, requires_grad=True)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """Compute KL divergence loss with batched forward passes."""
        batch = self._build_batch(inputs, model)
        if batch is None:
            logger.warning("Empty batch after building — returning zero loss")
            zero = self._zero_loss()
            return (zero, None) if return_outputs else zero

        with torch.no_grad():
            with disable_adapters(model):
                teacher_outputs = model(**batch["teacher_fwd"])
        student_outputs = model(**batch["student_fwd"])

        loss = self._kl_from_logits(
            teacher_outputs.logits,
            student_outputs.logits,
            batch["teacher_mask_padded"],
            batch["student_mask_padded"],
        )
        if loss is None:
            logger.warning("No aligned logit pairs after slicing — returning zero loss")
            zero = self._zero_loss()
            return (zero, None) if return_outputs else zero
        return (loss, None) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Eval step.

        With sequential_eval=True, teacher logits are moved to CPU before the
        student forward pass to halve peak VRAM usage.
        """
        if not self.args.sequential_eval:
            raw = model.module if hasattr(model, "module") else model
            with torch.no_grad():
                loss = self.compute_loss(raw, inputs)
            return (loss.detach(), None, None)

        # Sequential: teacher → CPU offload → student
        batch = self._build_batch(inputs, model)
        if batch is None:
            return (self._zero_loss(), None, None)

        base = model.module if hasattr(model, "module") else model
        fwd_fn = type(base).forward
        with torch.no_grad():
            with disable_adapters(base):
                teacher_logits = fwd_fn(base, **batch["teacher_fwd"]).logits.cpu()
            torch.cuda.empty_cache()
            student_outputs = fwd_fn(base, **batch["student_fwd"])

        loss = self._kl_from_logits(
            teacher_logits.to(student_outputs.logits.device),
            student_outputs.logits,
            batch["teacher_mask_padded"],
            batch["student_mask_padded"],
        )
        if loss is None:
            return (self._zero_loss(), None, None)
        return (loss.detach(), None, None)

    def training_step(self, model, inputs, num_items_in_batch=None) -> torch.Tensor:
        """Generate trajectories on-the-fly if no precomputed responses.

        Trajectory mode requires the final turn to be `user` (we sample an
        assistant response from the teacher). Multi-turn datasets with
        explicit assistant turns skip this path and use the provided responses.
        """
        existing_responses = inputs.get("responses") or []
        if any(r for r in existing_responses):
            return super().training_step(model, inputs, num_items_in_batch)

        # Trajectory mode: need user_messages (legacy) or turns ending in user.
        _, turns_list, _ = self._normalize_batch(inputs)
        user_messages = []
        for turns in turns_list:
            if not turns:
                continue
            last = turns[-1]
            if last.get("role") != "user":
                logger.warning(
                    "Trajectory mode: skipping row whose last turn is not 'user'"
                )
                continue
            user_messages.append(last["content"])

        all_user_messages, all_responses = [], []
        _TRAJECTORY_WARN_THRESHOLD = 64
        if len(user_messages) * self.num_trajectories > _TRAJECTORY_WARN_THRESHOLD:
            logger.warning(
                "Generating %d trajectories (%d prompts x %d each) — "
                "consider reducing batch size or num_trajectories if OOM occurs",
                len(user_messages) * self.num_trajectories,
                len(user_messages),
                self.num_trajectories,
            )
        for user_msg in user_messages:
            for _ in range(self.num_trajectories):
                response = self._generate_trajectory(user_msg)
                if response.strip():
                    all_user_messages.append(user_msg)
                    all_responses.append(response)

        if not all_responses:
            logger.warning("No valid trajectories generated — returning zero loss")
            return self._zero_loss()

        # Feed back as legacy-shape batch.
        inputs["user_messages"] = all_user_messages
        inputs["responses"] = all_responses
        inputs.pop("turns", None)
        inputs.pop("prefix_messages", None)
        return super().training_step(model, inputs, num_items_in_batch)


class PromptBakingTrainer(ContextBakingTrainer):
    """Deprecated alias for ContextBakingTrainer.

    Bakery now generalizes prompt baking to arbitrary prefix contexts.
    Use ContextBakingTrainer going forward.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "PromptBakingTrainer is deprecated; use ContextBakingTrainer instead. "
            "The old name will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
