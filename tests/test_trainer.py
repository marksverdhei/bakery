"""Tests for PromptBakingTrainer using a tiny GPT-2 model with LoRA."""

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


def _make_trainer(prompts, responses=None, batch_size=1):
    """Create a PromptBakingTrainer with tiny GPT-2 + LoRA for testing."""
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
        num_trajectories=1,
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


def test_compute_loss_returns_scalar():
    """compute_loss returns a scalar tensor with gradient."""
    trainer = _make_trainer(
        prompts=["What is 2+2?"],
        responses=["The answer is 4."],
    )
    inputs = {
        "user_messages": ["What is 2+2?"],
        "responses": ["The answer is 4."],
    }
    loss = trainer.compute_loss(trainer.model, inputs)
    assert loss.dim() == 0
    assert loss.requires_grad
    assert loss.item() > 0


def test_compute_loss_empty_inputs():
    """compute_loss returns zero for empty inputs."""
    trainer = _make_trainer(prompts=["placeholder"], responses=["placeholder"])
    loss = trainer.compute_loss(trainer.model, {"user_messages": [], "responses": []})
    assert loss.item() == 0.0


def test_compute_loss_empty_responses():
    """compute_loss returns zero when all responses are whitespace."""
    trainer = _make_trainer(prompts=["placeholder"], responses=["placeholder"])
    loss = trainer.compute_loss(
        trainer.model,
        {"user_messages": ["hello"], "responses": ["   "]},
    )
    assert loss.item() == 0.0


def test_compute_loss_batch():
    """compute_loss handles multiple pairs in a batch (tests padding logic)."""
    trainer = _make_trainer(
        prompts=["Short question?", "A much longer and more detailed question here?"],
        responses=[
            "Short answer.",
            "A longer and more detailed response to test padding.",
        ],
        batch_size=2,
    )
    inputs = {
        "user_messages": [
            "Short question?",
            "A much longer and more detailed question here?",
        ],
        "responses": [
            "Short answer.",
            "A longer and more detailed response to test padding.",
        ],
    }
    loss = trainer.compute_loss(trainer.model, inputs)
    assert loss.dim() == 0
    assert loss.requires_grad
    assert loss.item() > 0


def test_compute_loss_return_outputs():
    """compute_loss with return_outputs=True returns (loss, None)."""
    trainer = _make_trainer(
        prompts=["What is AI?"],
        responses=["AI is artificial intelligence."],
    )
    inputs = {
        "user_messages": ["What is AI?"],
        "responses": ["AI is artificial intelligence."],
    }
    result = trainer.compute_loss(trainer.model, inputs, return_outputs=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0].dim() == 0
    assert result[1] is None


def test_loss_is_differentiable():
    """Loss can be backpropagated through to LoRA parameters."""
    trainer = _make_trainer(
        prompts=["Hello"],
        responses=["Hi there!"],
    )
    inputs = {
        "user_messages": ["Hello"],
        "responses": ["Hi there!"],
    }
    loss = trainer.compute_loss(trainer.model, inputs)
    loss.backward()

    has_grad = False
    for name, param in trainer.model.named_parameters():
        if param.requires_grad and param.grad is not None:
            has_grad = True
            break
    assert has_grad, "No LoRA parameters received gradients"


def test_prompt_length_cache():
    """Prompt lengths are cached and reused across compute_loss calls."""
    trainer = _make_trainer(
        prompts=["What is 2+2?"],
        responses=["The answer is 4."],
    )
    assert len(trainer._prompt_length_cache) == 0

    inputs = {
        "user_messages": ["What is 2+2?"],
        "responses": ["The answer is 4."],
    }
    trainer.compute_loss(trainer.model, inputs)
    assert "What is 2+2?" in trainer._prompt_length_cache
    t_len, s_len = trainer._prompt_length_cache["What is 2+2?"]
    assert isinstance(t_len, int) and t_len > 0
    assert isinstance(s_len, int) and s_len > 0

    # Second call with same prompt should reuse cache (no new entries)
    trainer.compute_loss(trainer.model, inputs)
    assert len(trainer._prompt_length_cache) == 1


def test_prompt_length_cache_multiple_prompts():
    """Cache accumulates entries for different prompts."""
    trainer = _make_trainer(
        prompts=["Q1", "Q2"],
        responses=["A1", "A2"],
        batch_size=2,
    )
    inputs = {
        "user_messages": ["Q1", "Q2"],
        "responses": ["A1", "A2"],
    }
    trainer.compute_loss(trainer.model, inputs)
    assert len(trainer._prompt_length_cache) == 2
    assert "Q1" in trainer._prompt_length_cache
    assert "Q2" in trainer._prompt_length_cache


# ---------------------------------------------------------------------------
# prediction_step
# ---------------------------------------------------------------------------


def test_prediction_step_returns_triple():
    """prediction_step returns (loss, None, None) for eval."""
    trainer = _make_trainer(
        prompts=["What is 2+2?"],
        responses=["The answer is 4."],
    )
    inputs = {
        "user_messages": ["What is 2+2?"],
        "responses": ["The answer is 4."],
    }
    result = trainer.prediction_step(trainer.model, inputs, prediction_loss_only=True)
    assert isinstance(result, tuple) and len(result) == 3
    assert result[1] is None
    assert result[2] is None


def test_prediction_step_loss_is_scalar():
    """prediction_step loss is a detached scalar."""
    trainer = _make_trainer(
        prompts=["Hello"],
        responses=["Hi!"],
    )
    inputs = {"user_messages": ["Hello"], "responses": ["Hi!"]}
    loss, _, _ = trainer.prediction_step(
        trainer.model, inputs, prediction_loss_only=True
    )
    assert loss.dim() == 0
    assert not loss.requires_grad


def test_prediction_step_empty_inputs():
    """prediction_step returns zero loss for empty inputs."""
    trainer = _make_trainer(prompts=["placeholder"], responses=["placeholder"])
    inputs = {"user_messages": [], "responses": []}
    loss, _, _ = trainer.prediction_step(
        trainer.model, inputs, prediction_loss_only=True
    )
    assert loss.item() == 0.0


def test_prediction_step_sequential_eval_returns_triple():
    """sequential_eval prediction_step also returns a (loss, None, None) triple."""
    from bakery.config import BakeryConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig as PeftLoraConfig, get_peft_model
    from bakery.data import create_dataset, prompt_baking_collator
    from bakery.trainer import PromptBakingTrainer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = CHAT_TEMPLATE

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model = get_peft_model(
        model,
        PeftLoraConfig(
            r=4, lora_alpha=8, target_modules=["c_attn"], task_type="CAUSAL_LM"
        ),
    )

    args = BakeryConfig(
        output_dir="/tmp/bakery_test",
        system_prompt="You are a helpful assistant.",
        num_trajectories=1,
        trajectory_length=16,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        logging_steps=1,
        report_to="none",
        use_cpu=True,
        sequential_eval=True,
    )
    trainer = PromptBakingTrainer(
        model=model,
        args=args,
        train_dataset=create_dataset(["Q"], ["A"]),
        processing_class=tokenizer,
        data_collator=prompt_baking_collator,
    )

    inputs = {"user_messages": ["Hello"], "responses": ["Hi there!"]}
    result = trainer.prediction_step(model, inputs, prediction_loss_only=True)
    assert isinstance(result, tuple) and len(result) == 3
    assert result[1] is None and result[2] is None
    loss = result[0]
    assert loss.dim() == 0


# ---------------------------------------------------------------------------
# _generate_trajectories_batched
# ---------------------------------------------------------------------------

def test_generate_trajectories_batched_returns_pairs():
    """Batched generation returns (user_msg, response) pairs."""
    trainer = _make_trainer(prompts=["Hello?"], responses=["Hi."])
    results = trainer._generate_trajectories_batched(["Hello?"], num_trajectories=1)
    assert isinstance(results, list)
    assert len(results) > 0
    for msg, resp in results:
        assert isinstance(msg, str)
        assert isinstance(resp, str)


def test_generate_trajectories_batched_msg_matches_input():
    """Each returned user_msg must be one of the original input messages."""
    trainer = _make_trainer(prompts=["Q1", "Q2"], responses=["A1", "A2"])
    results = trainer._generate_trajectories_batched(["Q1", "Q2"], num_trajectories=1)
    returned_msgs = {msg for msg, _ in results}
    assert returned_msgs.issubset({"Q1", "Q2"})


def test_generate_trajectories_batched_num_trajectories():
    """With num_trajectories=2 each message appears up to 2 times."""
    trainer = _make_trainer(prompts=["Hi?"], responses=["Hello!"])
    results = trainer._generate_trajectories_batched(["Hi?"], num_trajectories=2)
    # At most 2 results for 1 prompt * 2 trajectories
    assert len(results) <= 2


def test_generate_trajectories_batched_empty_messages():
    """Empty user_messages list returns empty results."""
    trainer = _make_trainer(prompts=["placeholder"], responses=["placeholder"])
    results = trainer._generate_trajectories_batched([], num_trajectories=2)
    assert results == []


def test_generate_trajectories_batched_filters_empty_responses():
    """Pairs with empty responses after strip() must be excluded."""
    trainer = _make_trainer(prompts=["placeholder"], responses=["placeholder"])
    results = trainer._generate_trajectories_batched(["Hi"], num_trajectories=1)
    for _, resp in results:
        assert resp.strip() != ""
