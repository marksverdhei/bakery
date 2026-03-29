"""Tests for PromptBakingTrainer.training_step."""

import logging

import torch
from unittest.mock import patch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig as PeftLoraConfig, get_peft_model

from bakery.config import BakeryConfig
from bakery.data import create_dataset, prompt_baking_collator
from bakery.trainer import PromptBakingTrainer


CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "{{ m['role'] }}: {{ m['content'] }}\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}assistant: {% endif %}"
)


def _make_trainer(prompts=None, responses=None, num_trajectories=1, batch_size=1):
    """Tiny GPT-2 + LoRA trainer for testing."""
    prompts = prompts or ["What is 2+2?"]
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = CHAT_TEMPLATE

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    peft_config = PeftLoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["c_attn"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    args = BakeryConfig(
        output_dir="/tmp/bakery_test",
        system_prompt="You are a helpful assistant.",
        num_trajectories=num_trajectories,
        trajectory_length=16,
        per_device_train_batch_size=batch_size,
        num_train_epochs=1,
        logging_steps=1,
        report_to="none",
        use_cpu=True,
    )
    dataset = create_dataset(prompts, responses)
    trainer = PromptBakingTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=prompt_baking_collator,
    )
    return trainer


# ---------------------------------------------------------------------------
# training_step: responses already present → delegates to super
# ---------------------------------------------------------------------------


def test_training_step_with_existing_responses_delegates_to_super():
    """When inputs already have responses, training_step calls super()."""
    trainer = _make_trainer()
    inputs = {
        "user_messages": ["What is 2+2?"],
        "responses": ["The answer is 4."],
    }
    with patch.object(
        PromptBakingTrainer.__bases__[0],
        "training_step",
        return_value=torch.tensor(1.5),
    ) as mock_super:
        loss = trainer.training_step(trainer.model, inputs)
    mock_super.assert_called_once()
    assert loss.item() == 1.5


# ---------------------------------------------------------------------------
# training_step: no responses → generates trajectories via _generate_trajectory
# ---------------------------------------------------------------------------


def test_training_step_generates_trajectories_when_no_responses():
    """When no responses present, training_step calls _generate_trajectory."""
    trainer = _make_trainer(num_trajectories=2)
    inputs = {"user_messages": ["Hello?"]}

    with (
        patch.object(
            trainer, "_generate_trajectory", return_value="Hi there!"
        ) as mock_gen,
        patch.object(
            PromptBakingTrainer.__bases__[0],
            "training_step",
            return_value=torch.tensor(0.5),
        ),
    ):
        trainer.training_step(trainer.model, inputs)

    # Called once per prompt × num_trajectories
    assert mock_gen.call_count == 2


def test_training_step_populates_inputs_with_generated_responses():
    """training_step fills inputs['responses'] before delegating to super."""
    trainer = _make_trainer(num_trajectories=1)
    inputs = {"user_messages": ["What colour is the sky?"]}

    captured_inputs = {}

    def fake_super(model, inputs_arg, num_items=None):
        captured_inputs.update(inputs_arg)
        return torch.tensor(0.0)

    with (
        patch.object(trainer, "_generate_trajectory", return_value="Blue."),
        patch.object(
            PromptBakingTrainer.__bases__[0],
            "training_step",
            side_effect=fake_super,
        ),
    ):
        trainer.training_step(trainer.model, inputs)

    assert "responses" in captured_inputs
    assert captured_inputs["responses"] == ["Blue."]


def test_training_step_filters_blank_trajectories():
    """Blank trajectories are filtered out before delegating to super."""
    trainer = _make_trainer(num_trajectories=3)
    inputs = {"user_messages": ["Question?"]}

    # Return blank for first two, valid for third
    side_effects = ["", "   ", "Valid answer."]

    with (
        patch.object(trainer, "_generate_trajectory", side_effect=side_effects),
        patch.object(
            PromptBakingTrainer.__bases__[0],
            "training_step",
            return_value=torch.tensor(0.0),
        ) as mock_super,
    ):
        trainer.training_step(trainer.model, inputs)

    call_inputs = mock_super.call_args[0][1]
    assert len(call_inputs["responses"]) == 1
    assert call_inputs["responses"][0] == "Valid answer."


def test_training_step_returns_zero_when_all_trajectories_blank():
    """All-blank trajectories → zero-loss tensor, super not called."""
    trainer = _make_trainer(num_trajectories=2)
    inputs = {"user_messages": ["Prompt"]}

    with (
        patch.object(trainer, "_generate_trajectory", return_value=""),
        patch.object(
            PromptBakingTrainer.__bases__[0],
            "training_step",
        ) as mock_super,
    ):
        loss = trainer.training_step(trainer.model, inputs)

    mock_super.assert_not_called()
    assert loss.item() == 0.0
    assert loss.requires_grad


# ---------------------------------------------------------------------------
# training_step: warning when many trajectories
# ---------------------------------------------------------------------------


def test_training_step_warns_when_many_trajectories(caplog):
    """Logs a warning when total trajectories exceed threshold (64)."""
    # 9 prompts × 8 trajectories = 72 > 64 threshold
    prompts = [f"p{i}" for i in range(9)]
    trainer = _make_trainer(prompts=prompts, num_trajectories=8)
    inputs = {"user_messages": prompts}

    with (
        patch.object(trainer, "_generate_trajectory", return_value="ok"),
        patch.object(
            PromptBakingTrainer.__bases__[0],
            "training_step",
            return_value=torch.tensor(0.0),
        ),
        caplog.at_level(logging.WARNING),
    ):
        trainer.training_step(trainer.model, inputs)

    assert any("trajectories" in r.message.lower() for r in caplog.records)


def test_training_step_no_warn_below_threshold(caplog):
    """Does NOT warn when total trajectories are at or below 64."""
    # 4 prompts × 16 trajectories = 64, exactly at threshold, no warning
    prompts = [f"p{i}" for i in range(4)]
    trainer = _make_trainer(prompts=prompts, num_trajectories=16)
    inputs = {"user_messages": prompts}

    with (
        patch.object(trainer, "_generate_trajectory", return_value="ok"),
        patch.object(
            PromptBakingTrainer.__bases__[0],
            "training_step",
            return_value=torch.tensor(0.0),
        ),
        caplog.at_level(logging.WARNING),
    ):
        trainer.training_step(trainer.model, inputs)

    assert not any("trajectories" in r.message.lower() for r in caplog.records)
