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

from bakery.kl import compute_kl_divergence, disable_adapters


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
        return self.processing_class(text, add_special_tokens=False, **kwargs)

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

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute KL divergence loss with batched forward passes."""
        user_messages = inputs.get("user_messages", [])
        responses = inputs.get("responses", [])

        if not user_messages or not responses:
            loss = torch.tensor(0.0, device=model.device, requires_grad=True)
            return (loss, None) if return_outputs else loss

        pairs = [
            (msg, resp) for msg, resp in zip(user_messages, responses) if resp.strip()
        ]
        if not pairs:
            loss = torch.tensor(0.0, device=model.device, requires_grad=True)
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
            t_prompt = self._format_prompted(user_msg)
            teacher_prompt_lengths.append(
                self._tokenize(t_prompt, return_tensors="pt")["input_ids"].shape[1]
            )

            s_msgs = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": response},
            ]
            student_texts.append(
                self.processing_class.apply_chat_template(s_msgs, tokenize=False)
            )
            s_prompt = self._format_unprompted(user_msg)
            student_prompt_lengths.append(
                self._tokenize(s_prompt, return_tensors="pt")["input_ids"].shape[1]
            )

        teacher_inputs = self._tokenize(
            teacher_texts, return_tensors="pt", padding=True
        ).to(model.device)
        student_inputs = self._tokenize(
            student_texts, return_tensors="pt", padding=True
        ).to(model.device)

        teacher_token_type_ids = torch.zeros_like(teacher_inputs["input_ids"])
        student_token_type_ids = torch.zeros_like(student_inputs["input_ids"])

        with torch.no_grad():
            with disable_adapters(model):
                teacher_outputs = model(
                    input_ids=teacher_inputs["input_ids"],
                    attention_mask=teacher_inputs["attention_mask"],
                    token_type_ids=teacher_token_type_ids,
                )

        student_outputs = model(
            input_ids=student_inputs["input_ids"],
            attention_mask=student_inputs["attention_mask"],
            token_type_ids=student_token_type_ids,
        )

        total_loss = torch.tensor(0.0, device=model.device, requires_grad=True)
        count = 0

        for i in range(len(pairs)):
            t_start = teacher_prompt_lengths[i]
            s_start = student_prompt_lengths[i]

            t_logits = teacher_outputs.logits[i : i + 1, t_start - 1 : -1, :]
            s_logits = student_outputs.logits[i : i + 1, s_start - 1 : -1, :]

            t_mask = teacher_inputs["attention_mask"][i : i + 1, t_start:]
            s_mask = student_inputs["attention_mask"][i : i + 1, s_start:]

            min_len = min(t_logits.shape[1], s_logits.shape[1])
            if min_len == 0:
                continue

            t_logits = t_logits[:, :min_len, :]
            s_logits = s_logits[:, :min_len, :]
            mask = (t_mask[:, :min_len] * s_mask[:, :min_len]).float()

            loss = compute_kl_divergence(
                t_logits.detach(), s_logits, mask, self.kl_temperature
            )
            total_loss = total_loss + loss
            count += 1

        if count > 0:
            total_loss = total_loss / count

        return (total_loss, None) if return_outputs else total_loss

    def training_step(self, model, inputs, num_items_in_batch=None) -> torch.Tensor:
        """Generate trajectories on-the-fly if no precomputed responses."""
        existing_responses = inputs.get("responses", [])
        if existing_responses:
            return super().training_step(model, inputs, num_items_in_batch)

        user_messages = inputs.get("user_messages", [])
        all_user_messages, all_responses = [], []

        for user_msg in user_messages:
            for _ in range(self.num_trajectories):
                response = self._generate_trajectory(user_msg)
                if response.strip():
                    all_user_messages.append(user_msg)
                    all_responses.append(response)

        if not all_responses:
            return torch.tensor(0.0, device=model.device, requires_grad=True)

        inputs["user_messages"] = all_user_messages
        inputs["responses"] = all_responses
        return super().training_step(model, inputs, num_items_in_batch)
