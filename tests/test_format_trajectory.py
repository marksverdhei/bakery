"""Tests for PromptBakingTrainer format helpers and _generate_trajectory."""

import torch
from unittest.mock import patch, MagicMock
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


def _make_trainer(system_prompt="Be helpful.", num_trajectories=1):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = CHAT_TEMPLATE

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    peft_config = PeftLoraConfig(
        r=4, lora_alpha=8, target_modules=["c_attn"], task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    args = BakeryConfig(
        output_dir="/tmp/bakery_test",
        system_prompt=system_prompt,
        num_trajectories=num_trajectories,
        trajectory_length=16,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        logging_steps=1,
        report_to="none",
        use_cpu=True,
    )
    dataset = create_dataset(["Hello?"])
    trainer = PromptBakingTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=prompt_baking_collator,
    )
    return trainer


# ---------------------------------------------------------------------------
# _format_prompted
# ---------------------------------------------------------------------------

class TestFormatPrompted:
    def test_includes_system_prompt(self):
        trainer = _make_trainer(system_prompt="Custom system.")
        result = trainer._format_prompted("Hello?")
        assert "Custom system." in result

    def test_includes_user_message(self):
        trainer = _make_trainer()
        result = trainer._format_prompted("What is the capital of France?")
        assert "What is the capital of France?" in result

    def test_includes_generation_prompt(self):
        """apply_chat_template with add_generation_prompt=True appends 'assistant: '."""
        trainer = _make_trainer()
        result = trainer._format_prompted("Hello?")
        assert "assistant:" in result

    def test_returns_string(self):
        trainer = _make_trainer()
        assert isinstance(trainer._format_prompted("Hi"), str)


# ---------------------------------------------------------------------------
# _format_unprompted
# ---------------------------------------------------------------------------

class TestFormatUnprompted:
    def test_excludes_system_prompt(self):
        trainer = _make_trainer(system_prompt="System content here.")
        result = trainer._format_unprompted("Hello?")
        assert "System content here." not in result

    def test_includes_user_message(self):
        trainer = _make_trainer()
        result = trainer._format_unprompted("What time is it?")
        assert "What time is it?" in result

    def test_includes_generation_prompt(self):
        trainer = _make_trainer()
        result = trainer._format_unprompted("Hello?")
        assert "assistant:" in result

    def test_shorter_than_prompted(self):
        """Unprompted text should be shorter than prompted (no system msg)."""
        trainer = _make_trainer(system_prompt="A fairly long system prompt.")
        msg = "Hello?"
        assert len(trainer._format_unprompted(msg)) < len(trainer._format_prompted(msg))


# ---------------------------------------------------------------------------
# _generate_trajectory
# ---------------------------------------------------------------------------

class TestGenerateTrajectory:
    def test_returns_string(self):
        trainer = _make_trainer()
        result = trainer._generate_trajectory("What is 2+2?")
        assert isinstance(result, str)

    def test_strips_whitespace(self):
        """_generate_trajectory must strip leading/trailing whitespace."""
        trainer = _make_trainer()
        # Patch decode to return padded string
        with patch.object(
            trainer.processing_class,
            "decode",
            return_value="  Some response.  ",
        ):
            result = trainer._generate_trajectory("Q?")
        assert result == "Some response."

    def test_model_switched_to_eval_during_generation(self):
        """Model is set to eval() during generation."""
        trainer = _make_trainer()
        trainer.model.train()
        eval_modes = []

        original_generate = trainer.model.generate

        def spy_generate(**kwargs):
            eval_modes.append(trainer.model.training)
            return original_generate(**kwargs)

        with patch.object(trainer.model, "generate", side_effect=spy_generate):
            trainer._generate_trajectory("Q?")

        assert eval_modes and eval_modes[0] is False

    def test_model_restored_to_train_after_generation(self):
        """Model is restored to train() after generation."""
        trainer = _make_trainer()
        trainer.model.train()
        trainer._generate_trajectory("Q?")
        assert trainer.model.training

    def test_model_stays_eval_if_was_eval(self):
        """If model was already in eval(), it stays in eval() after generation."""
        trainer = _make_trainer()
        trainer.model.eval()
        trainer._generate_trajectory("Q?")
        assert not trainer.model.training

    def test_adapters_disabled_during_generation(self):
        """_generate_trajectory uses disable_adapters context manager."""
        trainer = _make_trainer()
        with patch("bakery.trainer.disable_adapters") as mock_ctx:
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=None)
            mock_cm.__exit__ = MagicMock(return_value=False)
            mock_ctx.return_value = mock_cm
            trainer._generate_trajectory("Q?")
        mock_ctx.assert_called_once()
