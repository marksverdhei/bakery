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

    # -- Loss computation --

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """Compute KL divergence loss with batched forward passes."""
        user_messages = inputs.get("user_messages", [])
        responses = inputs.get("responses", [])

        if not user_messages or not responses:
            logger.warning(
                "Batch has no user_messages or responses — returning zero loss"
            )
            loss = torch.tensor(0.0, device=self.args.device, requires_grad=True)
            return (loss, None) if return_outputs else loss

        pairs = [
            (msg, resp) for msg, resp in zip(user_messages, responses) if resp.strip()
        ]
        if not pairs:
            logger.warning(
                "All responses in batch are empty/whitespace — returning zero loss"
            )
            loss = torch.tensor(0.0, device=self.args.device, requires_grad=True)
            return (loss, None) if return_outputs else loss

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

        teacher_fwd = dict(
            input_ids=teacher_inputs["input_ids"],
            attention_mask=teacher_inputs["attention_mask"],
        )
        student_fwd = dict(
            input_ids=student_inputs["input_ids"],
            attention_mask=student_inputs["attention_mask"],
        )
        # Some architectures (e.g. Gemma 3) require token_type_ids during training.
        # The tokenizer may not return them, so create zeros if the model expects them.
        if hasattr(model.config, "model_type") and model.config.model_type in (
            "gemma3",
        ):
            teacher_fwd["token_type_ids"] = torch.zeros_like(
                teacher_inputs["input_ids"]
            )
            student_fwd["token_type_ids"] = torch.zeros_like(
                student_inputs["input_ids"]
            )
        elif "token_type_ids" in teacher_inputs:
            teacher_fwd["token_type_ids"] = teacher_inputs["token_type_ids"]
            student_fwd["token_type_ids"] = student_inputs["token_type_ids"]

        with torch.no_grad():
            with disable_adapters(model):
                teacher_outputs = model(**teacher_fwd)

        student_outputs = model(**student_fwd)

        # With left-padding, each sequence has leading pad tokens that shift
        # the real content rightward. Compute per-sequence padding offsets.
        t_seq_len = teacher_inputs["input_ids"].shape[1]
        s_seq_len = student_inputs["input_ids"].shape[1]
        t_real_lengths = teacher_inputs["attention_mask"].sum(dim=1)
        s_real_lengths = student_inputs["attention_mask"].sum(dim=1)
        B = len(pairs)
        V = teacher_outputs.logits.shape[-1]

        # Compute per-sample response start positions (in logit space, shifted -1
        # so that logit[t] predicts token[t+1]).
        t_starts = [
            int(t_seq_len - t_real_lengths[i].item()) + teacher_prompt_lengths[i]
            for i in range(B)
        ]
        s_starts = [
            int(s_seq_len - s_real_lengths[i].item()) + student_prompt_lengths[i]
            for i in range(B)
        ]
        # Response length for sample i: from start to seq_end (exclusive), capped
        # at the other sequence's response length to keep teacher/student aligned.
        t_resp_lens = [t_seq_len - 1 - (t_starts[i] - 1) for i in range(B)]
        s_resp_lens = [s_seq_len - 1 - (s_starts[i] - 1) for i in range(B)]
        min_resp_lens = [min(t_resp_lens[i], s_resp_lens[i]) for i in range(B)]

        # Filter out zero-length samples (degenerate prompts/responses).
        valid = [i for i, L in enumerate(min_resp_lens) if L > 0]
        if not valid:
            logger.warning("No aligned logit pairs after slicing — returning zero loss")
            zero = torch.tensor(0.0, device=self.args.device, requires_grad=True)
            return (zero, None) if return_outputs else zero

        max_resp_len = max(min_resp_lens[i] for i in valid)

        # Build batched logit tensors [|valid|, max_resp_len, V] by copying each
        # sample's response slice. This CPU loop is cheap (shapes only differ in
        # sequence position); the expensive softmax/KL runs once on the batch.
        t_batch = teacher_outputs.logits.new_zeros(len(valid), max_resp_len, V)
        s_batch = student_outputs.logits.new_zeros(len(valid), max_resp_len, V)
        mask_batch = teacher_outputs.logits.new_zeros(len(valid), max_resp_len)

        for out_idx, i in enumerate(valid):
            L = min_resp_lens[i]
            ts = t_starts[i] - 1  # logit position for first response token
            ss = s_starts[i] - 1
            t_batch[out_idx, :L] = teacher_outputs.logits[i, ts : ts + L]
            s_batch[out_idx, :L] = student_outputs.logits[i, ss : ss + L]
            mask_batch[out_idx, :L] = 1.0

        per_sample_losses = compute_kl_divergence(
            t_batch.detach(), s_batch, mask_batch, self.kl_temperature,
            per_sample=True,
        )
        total_loss = per_sample_losses.mean()
        return (total_loss, None) if return_outputs else total_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Eval step: reuse compute_loss so the collated batch format works.

        With sequential_eval=True, teacher logits are moved to CPU before the
        student forward pass to halve peak VRAM usage.
        """
        if not self.args.sequential_eval:
            # Unwrap to bypass Accelerate's fp32 upcast wrapper
            raw = model.module if hasattr(model, "module") else model
            with torch.no_grad():
                loss = self.compute_loss(raw, inputs)
            return (loss.detach(), None, None)

        # Sequential eval: teacher → offload logits to CPU → student
        user_messages = inputs.get("user_messages", [])
        responses = inputs.get("responses", [])
        pairs = [(m, r) for m, r in zip(user_messages, responses) if r.strip()]
        if not pairs:
            return (
                torch.tensor(0.0, device=self.args.device, requires_grad=True),
                None,
                None,
            )

        teacher_texts, student_texts = [], []
        teacher_prompt_lengths, student_prompt_lengths = [], []
        for user_msg, response in pairs:
            t_msgs = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": response},
            ]
            s_msgs = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": response},
            ]
            teacher_texts.append(
                self.processing_class.apply_chat_template(t_msgs, tokenize=False)
            )
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
            if hasattr(model.config, "model_type") and model.config.model_type in (
                "gemma3",
            ):
                fwd["token_type_ids"] = torch.zeros_like(tok_inputs["input_ids"])
            elif "token_type_ids" in tok_inputs:
                fwd["token_type_ids"] = tok_inputs["token_type_ids"]
            return fwd

        # Accelerate replaces model.forward with a wrapper that upcasts to fp32.
        # Bypass by calling the CLASS forward method directly.
        base = model.module if hasattr(model, "module") else model
        fwd_fn = type(base).forward
        with torch.no_grad():
            with disable_adapters(base):
                teacher_logits = fwd_fn(base, **_make_fwd(teacher_inputs)).logits.cpu()
            torch.cuda.empty_cache()
            student_outputs = fwd_fn(base, **_make_fwd(student_inputs))

        t_seq_len = teacher_inputs["input_ids"].shape[1]
        s_seq_len = student_inputs["input_ids"].shape[1]
        t_real_lengths = teacher_inputs["attention_mask"].sum(dim=1)
        s_real_lengths = student_inputs["attention_mask"].sum(dim=1)
        B = len(pairs)
        V = teacher_logits.shape[-1]

        t_starts = [
            int(t_seq_len - t_real_lengths[i].item()) + teacher_prompt_lengths[i]
            for i in range(B)
        ]
        s_starts = [
            int(s_seq_len - s_real_lengths[i].item()) + student_prompt_lengths[i]
            for i in range(B)
        ]
        t_resp_lens = [t_seq_len - 1 - (t_starts[i] - 1) for i in range(B)]
        s_resp_lens = [s_seq_len - 1 - (s_starts[i] - 1) for i in range(B)]
        min_resp_lens = [min(t_resp_lens[i], s_resp_lens[i]) for i in range(B)]

        valid = [i for i, L in enumerate(min_resp_lens) if L > 0]
        if not valid:
            return (
                torch.tensor(0.0, device=self.args.device, requires_grad=True),
                None,
                None,
            )

        max_resp_len = max(min_resp_lens[i] for i in valid)
        dev = student_outputs.logits.device
        t_batch = student_outputs.logits.new_zeros(len(valid), max_resp_len, V)
        s_batch = student_outputs.logits.new_zeros(len(valid), max_resp_len, V)
        mask_batch = student_outputs.logits.new_zeros(len(valid), max_resp_len)

        for out_idx, i in enumerate(valid):
            L = min_resp_lens[i]
            ts = t_starts[i] - 1
            ss = s_starts[i] - 1
            t_batch[out_idx, :L] = teacher_logits[i, ts : ts + L].to(dev)
            s_batch[out_idx, :L] = student_outputs.logits[i, ss : ss + L]
            mask_batch[out_idx, :L] = 1.0

        per_sample_losses = compute_kl_divergence(
            t_batch, s_batch, mask_batch, self.kl_temperature, per_sample=True
        )
        return (per_sample_losses.mean().detach(), None, None)

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
