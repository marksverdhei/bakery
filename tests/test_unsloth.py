"""Unsloth integration tests.

Validates that Unsloth-patched models correctly support adapter toggling,
which is critical for bakery's teacher-student KL distillation pattern.

Run with: uv run pytest tests/test_unsloth.py -v -s
Requires: GPU with CUDA, unsloth package installed.
"""

import pytest
import torch

pytestmark = [pytest.mark.gpu]


def _has_unsloth():
    try:
        import unsloth  # noqa: F401

        return True
    except ImportError:
        return False


def _has_gpu():
    return torch.cuda.is_available()


requires_unsloth = pytest.mark.skipif(
    not _has_unsloth() or not _has_gpu(),
    reason="Requires unsloth and CUDA GPU",
)


@requires_unsloth
def test_unsloth_adapter_toggling():
    """CRITICAL: Verify logits differ between adapters enabled and disabled.

    If this test fails, Unsloth's custom kernels do not respect
    disable_adapter_layers(), and bakery's teacher-student distillation
    would be silently broken.
    """
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen3-0.6B",
        max_seq_length=512,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0,
        bias="none",
    )

    # Perturb LoRA weights so they produce a measurable difference
    for name, param in model.named_parameters():
        if "lora" in name.lower() and param.requires_grad:
            param.data.add_(torch.randn_like(param) * 0.1)

    text = "What is the capital of France?"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    model.eval()

    # Forward with adapters enabled (student mode)
    with torch.no_grad():
        logits_with_adapter = model(**inputs).logits.clone()

    # Forward with adapters disabled (teacher mode)
    with torch.no_grad():
        model.disable_adapter_layers()
        logits_without_adapter = model(**inputs).logits.clone()
        model.enable_adapter_layers()

    # Logits MUST differ — if they don't, adapter toggling is broken
    diff = (logits_with_adapter - logits_without_adapter).abs().max().item()
    assert diff > 1e-3, (
        f"Adapter toggling has no effect on logits (max diff={diff:.6f}). "
        "Unsloth's custom kernels may not respect disable_adapter_layers(). "
        "DO NOT use Unsloth with bakery until this is fixed."
    )


@requires_unsloth
def test_unsloth_gradient_flow():
    """Verify gradients flow through LoRA params on an Unsloth-patched model."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen3-0.6B",
        max_seq_length=512,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0,
        bias="none",
    )

    text = "What is the capital of France?"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    model.train()
    outputs = model(**inputs)
    loss = outputs.logits.sum()
    loss.backward()

    lora_params_with_grad = 0
    lora_params_total = 0
    for name, param in model.named_parameters():
        if "lora" in name.lower() and param.requires_grad:
            lora_params_total += 1
            if param.grad is not None and param.grad.abs().sum() > 0:
                lora_params_with_grad += 1

    assert lora_params_with_grad > 0, "No LoRA parameters received gradients"
    # With gradient checkpointing and short sequences, not all params
    # may get non-zero gradients. At least half should.
    ratio = lora_params_with_grad / lora_params_total
    assert ratio > 0.25, (
        f"Only {lora_params_with_grad}/{lora_params_total} LoRA params "
        f"got gradients ({ratio:.0%})"
    )


@requires_unsloth
def test_unsloth_compute_loss_integration():
    """Test full compute_loss with Unsloth model and adapter toggling."""
    from unsloth import FastLanguageModel

    from bakery.config import BakeryConfig
    from bakery.data import prompt_baking_collator
    from bakery.trainer import PromptBakingTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen3-0.6B",
        max_seq_length=512,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0,
        bias="none",
    )

    args = BakeryConfig(
        output_dir="/tmp/bakery_unsloth_test",
        system_prompt="You are a helpful assistant.",
        num_trajectories=1,
        trajectory_length=16,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        logging_steps=1,
        report_to="none",
        gradient_checkpointing=True,
        bf16=True,
    )

    prompts = ["What is 2+2?"]
    responses = ["The answer is 4."]
    # Use a minimal list-based dataset to avoid Python 3.14 pickle
    # incompatibility with HF datasets (Pickler._batch_setitems).
    dummy = [{"user_messages": "x", "responses": "y"}]

    trainer = PromptBakingTrainer(
        model=model,
        args=args,
        train_dataset=dummy,
        processing_class=tokenizer,
        data_collator=prompt_baking_collator,
    )

    inputs = {"user_messages": prompts, "responses": responses}
    loss = trainer.compute_loss(model, inputs)

    assert loss.item() > 0, "Loss should be positive"
    assert loss.requires_grad, "Loss must have gradients"

    loss.backward()

    lora_grads = sum(
        1
        for n, p in model.named_parameters()
        if "lora" in n.lower() and p.requires_grad and p.grad is not None
    )
    assert lora_grads > 0, "No LoRA params got gradients through compute_loss"
