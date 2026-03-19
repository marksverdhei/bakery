"""Prompt baking trainer via KL divergence.

Inherits from transformers.Trainer to get standard HF training infrastructure
(logging, checkpointing, schedulers, gradient accumulation) while implementing
the prompt baking training loop:

- Single model with PEFT adapter toggling (no duplicate weights)
- Teacher: adapters disabled, sees system prompt
- Student: adapters enabled, no system prompt
- Per-token masked KL divergence loss
- On-the-fly trajectory generation from teacher

Based on the Prompt Baking paper (arxiv 2409.13697).
"""

import logging
import torch
from typing import Optional, Callable

from datasets import Dataset
from transformers import (
    Trainer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    GenerationConfig,
)
from transformers.trainer_utils import EvalPrediction

from bakery.kl import compute_kl_divergence, disable_adapters, padding_side

logger = logging.getLogger(__name__)


class PromptBakingTrainer(Trainer):
    """Trainer that bakes system prompts into model weights via KL divergence.

    Overrides compute_loss() and training_step() to inject prompt baking logic:
    1. training_step: generates trajectories from teacher (adapters disabled)
    2. compute_loss: computes per-token KL divergence between teacher and student
    """

    def __init__(
        self,
        model: PreTrainedModel | str | None = None,
        args=None,
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
        self.system_prompt = args.system_prompt
        self.num_trajectories = args.num_trajectories
        self.trajectory_length = args.trajectory_length
        self.sampling_temperature = args.sampling_temperature
        self.kl_temperature = args.temperature

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

        self._prompt_length_cache: dict[str, tuple[int, int]] = {}

        self.model_accepts_loss_kwargs = False
        self.generation_config = GenerationConfig(
            max_new_tokens=self.trajectory_length,
            temperature=self.sampling_temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=self.processing_class.pad_token_id,
        )

    # -- Chat formatting --

    def _tokenize(self, text: str, **kwargs) -> dict:
        max_len = getattr(self.args, "max_seq_length", None)
        if max_len:
            kwargs.setdefault("truncation", True)
            kwargs.setdefault("max_length", max_len)
        return self.processing_class(text, add_special_tokens=False, **kwargs)

    def _get_prompt_lengths(self, user_message: str) -> tuple[int, int]:
        """Return (teacher_prompt_length, student_prompt_length) for a user message.

        Results are cached because prompt lengths are deterministic for a given
        user_message (the system_prompt is constant across the trainer's lifetime).
        """
        if user_message not in self._prompt_length_cache:
            t_prompt = self._format_prompted(user_message)
            t_len = self._tokenize(t_prompt, return_tensors="pt")["input_ids"].shape[1]
            s_prompt = self._format_unprompted(user_message)
            s_len = self._tokenize(s_prompt, return_tensors="pt")["input_ids"].shape[1]
            self._prompt_length_cache[user_message] = (t_len, s_len)
        return self._prompt_length_cache[user_message]

    def _format_prompted(self, user_message: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        return self.processing_class.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _format_unprompted(self, user_message: str) -> str:
        messages = [{"role": "user", "content": user_message}]
        return self.processing_class.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # -- Trajectory generation --

    def _generate_trajectory(self, user_message: str) -> str:
        prompt = self._format_prompted(user_message)
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

    # -- Shared helpers --

    def _prepare_inputs(self, pairs, model):
        """Format, tokenize, and build forward-pass dicts for teacher/student.

        Returns:
            (teacher_inputs, student_inputs, teacher_fwd, student_fwd,
             teacher_prompt_lengths, student_prompt_lengths)
        """
        teacher_texts, student_texts = [], []
        teacher_prompt_lengths, student_prompt_lengths = [], []

        for user_msg, response in pairs:
            t_msgs = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": response},
            ]
            teacher_texts.append(
                self.processing_class.apply_chat_template(t_msgs, tokenize=False)
            )

            s_msgs = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": response},
            ]
            student_texts.append(
                self.processing_class.apply_chat_template(s_msgs, tokenize=False)
            )

            t_len, s_len = self._get_prompt_lengths(user_msg)
            teacher_prompt_lengths.append(t_len)
            student_prompt_lengths.append(s_len)

        with padding_side(self.processing_class, "left"):
            teacher_inputs = self._tokenize(
                teacher_texts, return_tensors="pt", padding=True
            ).to(model.device)
            student_inputs = self._tokenize(
                student_texts, return_tensors="pt", padding=True
            ).to(model.device)

        def _make_fwd(tok_inputs):
            fwd = dict(
                input_ids=tok_inputs["input_ids"],
                attention_mask=tok_inputs["attention_mask"],
            )
            # Some architectures (e.g. Gemma 3) require token_type_ids during training.
            if hasattr(model.config, "model_type") and model.config.model_type in (
                "gemma3",
            ):
                fwd["token_type_ids"] = torch.zeros_like(tok_inputs["input_ids"])
            elif "token_type_ids" in tok_inputs:
                fwd["token_type_ids"] = tok_inputs["token_type_ids"]
            return fwd

        teacher_fwd = _make_fwd(teacher_inputs)
        student_fwd = _make_fwd(student_inputs)

        return (
            teacher_inputs,
            student_inputs,
            teacher_fwd,
            student_fwd,
            teacher_prompt_lengths,
            student_prompt_lengths,
        )

    def _per_sample_kl(
        self,
        pairs,
        teacher_logits,
        student_logits,
        teacher_inputs,
        student_inputs,
        teacher_prompt_lengths,
        student_prompt_lengths,
    ) -> list[torch.Tensor]:
        """Compute per-sample KL divergence losses from aligned logit slices.

        Handles left-padding offsets and prompt-length masking.
        """
        t_seq_len = teacher_inputs["input_ids"].shape[1]
        s_seq_len = student_inputs["input_ids"].shape[1]
        t_real_lengths = teacher_inputs["attention_mask"].sum(dim=1)
        s_real_lengths = student_inputs["attention_mask"].sum(dim=1)

        losses = []
        for i in range(len(pairs)):
            t_pad = t_seq_len - t_real_lengths[i].item()
            s_pad = s_seq_len - s_real_lengths[i].item()
            t_start = int(t_pad) + teacher_prompt_lengths[i]
            s_start = int(s_pad) + student_prompt_lengths[i]

            # Logits at position t predict token t+1, so shift back by 1.
            # Slice to -1 because the last position predicts beyond the sequence.
            t_log = teacher_logits[i : i + 1, t_start - 1 : -1, :]
            s_log = student_logits[i : i + 1, s_start - 1 : -1, :]

            t_mask = teacher_inputs["attention_mask"][i : i + 1, t_start:]
            s_mask = student_inputs["attention_mask"][i : i + 1, s_start:]

            min_len = min(t_log.shape[1], s_log.shape[1])
            if min_len == 0:
                continue

            t_log = t_log[:, :min_len, :]
            s_log = s_log[:, :min_len, :]
            mask = (t_mask[:, :min_len] * s_mask[:, :min_len]).float()

            losses.append(
                compute_kl_divergence(t_log.detach(), s_log, mask, self.kl_temperature)
            )
        return losses

    def _zero_loss(self):
        """Return a zero loss tensor that supports backward()."""
        return torch.tensor(0.0, device=self.args.device, requires_grad=True)

    def _validate_batch(self, inputs):
        """Extract and validate (user_message, response) pairs from a batch.

        Returns:
            pairs or None if batch is empty/invalid (with warnings logged).
        """
        user_messages = inputs.get("user_messages", [])
        responses = inputs.get("responses", [])

        if not user_messages or not responses:
            logger.warning(
                "Batch has no user_messages or responses — returning zero loss"
            )
            return None

        pairs = [
            (msg, resp) for msg, resp in zip(user_messages, responses) if resp.strip()
        ]
        if not pairs:
            logger.warning(
                "All responses in batch are empty/whitespace — returning zero loss"
            )
            return None

        return pairs

    # -- Loss computation --

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """Compute KL divergence loss with batched forward passes."""
        pairs = self._validate_batch(inputs)
        if pairs is None:
            loss = self._zero_loss()
            return (loss, None) if return_outputs else loss

        (
            teacher_inputs,
            student_inputs,
            teacher_fwd,
            student_fwd,
            teacher_prompt_lengths,
            student_prompt_lengths,
        ) = self._prepare_inputs(pairs, model)

        with torch.no_grad():
            with disable_adapters(model):
                teacher_outputs = model(**teacher_fwd)

        student_outputs = model(**student_fwd)

        losses = self._per_sample_kl(
            pairs,
            teacher_outputs.logits,
            student_outputs.logits,
            teacher_inputs,
            student_inputs,
            teacher_prompt_lengths,
            student_prompt_lengths,
        )

        if not losses:
            logger.warning("No aligned logit pairs after slicing — returning zero loss")
            zero = self._zero_loss()
            return (zero, None) if return_outputs else zero

        total_loss = torch.stack(losses).mean()
        return (total_loss, None) if return_outputs else total_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Eval step: reuse compute_loss so the collated batch format works.

        With sequential_eval=True, teacher logits are moved to CPU before the
        student forward pass to halve peak VRAM usage.
        """
        if not self.args.sequential_eval:
            raw = model.module if hasattr(model, "module") else model
            with torch.no_grad():
                loss = self.compute_loss(raw, inputs)
            return (loss.detach(), None, None)

        # Sequential eval: teacher → offload logits to CPU → student
        pairs = self._validate_batch(inputs)
        if pairs is None:
            return (self._zero_loss(), None, None)

        (
            teacher_inputs,
            student_inputs,
            teacher_fwd,
            student_fwd,
            teacher_prompt_lengths,
            student_prompt_lengths,
        ) = self._prepare_inputs(pairs, model)

        # Accelerate replaces model.forward with a wrapper that upcasts to fp32.
        # Bypass by calling the CLASS forward method directly.
        base = model.module if hasattr(model, "module") else model
        fwd_fn = type(base).forward
        with torch.no_grad():
            with disable_adapters(base):
                teacher_logits = fwd_fn(base, **teacher_fwd).logits.cpu()
            torch.cuda.empty_cache()
            student_logits = fwd_fn(base, **student_fwd).logits

        losses = self._per_sample_kl(
            pairs,
            teacher_logits.to(model.device),
            student_logits,
            teacher_inputs,
            student_inputs,
            teacher_prompt_lengths,
            student_prompt_lengths,
        )

        if not losses:
            return (self._zero_loss(), None, None)
        return (torch.stack(losses).mean().detach(), None, None)

    def training_step(self, model, inputs, num_items_in_batch=None) -> torch.Tensor:
        """Generate trajectories on-the-fly if no precomputed responses."""
        existing_responses = inputs.get("responses", [])
        if existing_responses:
            return super().training_step(model, inputs, num_items_in_batch)

        user_messages = inputs.get("user_messages", [])
        all_user_messages, all_responses = [], []

        # Threshold beyond which memory usage from accumulated trajectories
        # may become significant (each trajectory requires a full forward pass).
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
            return torch.tensor(0.0, device=self.args.device, requires_grad=True)

        inputs["user_messages"] = all_user_messages
        inputs["responses"] = all_responses
        return super().training_step(model, inputs, num_items_in_batch)
