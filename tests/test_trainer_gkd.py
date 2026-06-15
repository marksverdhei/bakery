"""Integration tests: ContextBakingTrainer with an external teacher backend.

Uses a tiny shared-tokenizer pair (gpt2 student + a fresh gpt2 teacher) to
validate the GKD code path end-to-end on CPU.
"""

import pytest
import torch
from peft import LoraConfig as PeftLoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from bakery.config import BakeryConfig, ContextConfig
from bakery.data import prompt_baking_collator
from bakery.masking import clear_mask_cache
from bakery.teachers import HFTeacher
from bakery.trainer import ContextBakingTrainer

CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "{{ m['role'] }}: {{ m['content'] }}\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}assistant: {% endif %}"
)


def _mk_tokenizer():
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    tok.chat_template = CHAT_TEMPLATE
    return tok


@pytest.fixture(scope="module")
def gkd_trainer():
    """Tiny student (gpt2 + LoRA) + tiny teacher (separate gpt2) on CPU."""
    tokenizer = _mk_tokenizer()
    student = AutoModelForCausalLM.from_pretrained("gpt2")
    student = get_peft_model(
        student,
        PeftLoraConfig(
            r=4, lora_alpha=8, target_modules=["c_attn"], task_type="CAUSAL_LM"
        ),
    )

    args = BakeryConfig(
        output_dir="/tmp/bakery_gkd",
        num_trajectories=1,
        trajectory_length=8,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        logging_steps=1,
        report_to="none",
        use_cpu=True,
    )
    context_config = ContextConfig(
        prefix_messages=[{"role": "system", "content": "Be brief."}]
    )
    # A fresh gpt2 as the "teacher" — same tokenizer, different params (no LoRA).
    teacher = HFTeacher(model_name_or_path="gpt2", torch_dtype="float32")

    return ContextBakingTrainer(
        model=student,
        args=args,
        context_config=context_config,
        teacher_backend=teacher,
        teacher_top_k=16,
        processing_class=tokenizer,
        data_collator=prompt_baking_collator,
    )


# ---------- compute_loss with external teacher ----------


def test_gkd_compute_loss_returns_scalar(gkd_trainer):
    clear_mask_cache()
    loss = gkd_trainer.compute_loss(
        gkd_trainer.model,
        {"user_messages": ["What is two plus two?"], "responses": ["four."]},
    )
    assert loss.dim() == 0
    assert loss.item() >= 0


def test_gkd_compute_loss_nonzero_for_valid_batch(gkd_trainer):
    clear_mask_cache()
    loss = gkd_trainer.compute_loss(
        gkd_trainer.model,
        {
            "user_messages": ["What is two plus two?", "Capital of France?"],
            "responses": ["four.", "Paris."],
        },
    )
    assert loss.item() > 0


def test_gkd_compute_loss_zero_when_no_responses(gkd_trainer):
    clear_mask_cache()
    loss = gkd_trainer.compute_loss(
        gkd_trainer.model,
        {"user_messages": ["q"], "responses": [""]},
    )
    assert loss.item() == 0.0


def test_gkd_loss_is_differentiable_only_through_student(gkd_trainer):
    """Backprop should hit student LoRA params, not teacher."""
    clear_mask_cache()
    loss = gkd_trainer.compute_loss(
        gkd_trainer.model,
        {"user_messages": ["hi"], "responses": ["hey."]},
    )
    loss.backward()
    student_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in gkd_trainer.model.parameters()
        if p.requires_grad
    )
    assert student_has_grad
    # Teacher params should not have grads — they were registered with requires_grad=False.
    teacher_has_grad = any(
        p.grad is not None for p in gkd_trainer.teacher_backend.model.parameters()
    )
    assert not teacher_has_grad


def test_gkd_uses_topk_path(gkd_trainer):
    """Smoke: the teacher dispatch should pick the sparse path."""
    clear_mask_cache()
    payload = gkd_trainer._teacher_forward(
        gkd_trainer.model,
        gkd_trainer._build_batch(
            {"user_messages": ["q"], "responses": ["a."]}, gkd_trainer.model
        ),
    )
    assert payload[0] == "topk"
    assert payload[1].shape[-1] <= gkd_trainer.teacher_top_k


def test_gkd_prediction_step_returns_triple(gkd_trainer):
    """Eval path also takes the topk route and returns a detached scalar."""
    clear_mask_cache()
    out = gkd_trainer.prediction_step(
        gkd_trainer.model,
        {"user_messages": ["q"], "responses": ["a."]},
        prediction_loss_only=True,
    )
    assert isinstance(out, tuple) and len(out) == 3
    loss = out[0]
    assert loss.dim() == 0
    assert not loss.requires_grad


# ---------- generate via external teacher ----------


def test_gkd_generate_trajectory_uses_external_teacher(gkd_trainer):
    """When teacher_backend is set, _generate_trajectory delegates to it."""
    # The teacher's tokenizer needs a chat template for generate() — real
    # chat-tuned models (Gemma3-it, Qwen) ship one, gpt2 doesn't.
    gkd_trainer.teacher_backend.tokenizer.chat_template = CHAT_TEMPLATE
    out = gkd_trainer._generate_trajectory("Hello?")
    assert isinstance(out, str)


# ---------- local-toggle path still works (regression) ----------


# ---------- JSD wiring ----------


def test_gkd_jsd_beta_routes_to_jsd(gkd_trainer):
    """gkd_jsd_beta > 0 should route _kl_from_topk through topk_jsd."""
    clear_mask_cache()
    gkd_trainer.gkd_jsd_beta = 0.5
    try:
        loss = gkd_trainer.compute_loss(
            gkd_trainer.model,
            {"user_messages": ["hello"], "responses": ["world"]},
        )
        assert loss.item() >= 0
    finally:
        gkd_trainer.gkd_jsd_beta = 0.0


def test_gkd_jsd_beta_default_matches_forward_kl(gkd_trainer):
    """β=0 default and explicit β=0 should produce the same loss."""
    clear_mask_cache()
    inputs = {"user_messages": ["hello"], "responses": ["world"]}
    loss_default = gkd_trainer.compute_loss(gkd_trainer.model, inputs)
    gkd_trainer.gkd_jsd_beta = 0.0
    clear_mask_cache()
    loss_explicit = gkd_trainer.compute_loss(gkd_trainer.model, inputs)
    assert torch.allclose(loss_default, loss_explicit, atol=1e-5)


def test_trainer_rejects_invalid_jsd_beta():
    tokenizer = _mk_tokenizer()
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model = get_peft_model(
        model,
        PeftLoraConfig(
            r=4, lora_alpha=8, target_modules=["c_attn"], task_type="CAUSAL_LM"
        ),
    )
    with pytest.raises(ValueError, match="gkd_jsd_beta"):
        ContextBakingTrainer(
            model=model,
            args=BakeryConfig(
                output_dir="/tmp/bakery_bad",
                per_device_train_batch_size=1,
                num_train_epochs=1,
                logging_steps=1,
                report_to="none",
                use_cpu=True,
            ),
            context_config=ContextConfig(
                prefix_messages=[{"role": "system", "content": "s"}]
            ),
            gkd_jsd_beta=2.0,
            processing_class=tokenizer,
            data_collator=prompt_baking_collator,
        )


def test_trainer_rejects_invalid_on_policy_fraction():
    tokenizer = _mk_tokenizer()
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model = get_peft_model(
        model,
        PeftLoraConfig(
            r=4, lora_alpha=8, target_modules=["c_attn"], task_type="CAUSAL_LM"
        ),
    )
    with pytest.raises(ValueError, match="gkd_on_policy_fraction"):
        ContextBakingTrainer(
            model=model,
            args=BakeryConfig(
                output_dir="/tmp/bakery_bad2",
                per_device_train_batch_size=1,
                num_train_epochs=1,
                logging_steps=1,
                report_to="none",
                use_cpu=True,
            ),
            context_config=ContextConfig(
                prefix_messages=[{"role": "system", "content": "s"}]
            ),
            gkd_on_policy_fraction=-0.1,
            processing_class=tokenizer,
            data_collator=prompt_baking_collator,
        )


# ---------- on-policy routing ----------


def test_on_policy_fraction_one_always_samples_from_student(gkd_trainer, monkeypatch):
    """gkd_on_policy_fraction=1.0 → _generate_trajectory uses student sampler."""
    called = {"student": 0, "teacher": 0}
    monkeypatch.setattr(
        gkd_trainer,
        "_sample_from_student",
        lambda u: (
            called.__setitem__("student", called["student"] + 1),
            "student-sample",
        )[1],
    )
    monkeypatch.setattr(
        gkd_trainer.teacher_backend,
        "generate",
        lambda msgs, max_new_tokens=256: (
            called.__setitem__("teacher", called["teacher"] + 1),
            "teacher-sample",
        )[1],
    )
    gkd_trainer.gkd_on_policy_fraction = 1.0
    try:
        for _ in range(5):
            out = gkd_trainer._generate_trajectory("q?")
            assert out == "student-sample"
        assert called["student"] == 5
        assert called["teacher"] == 0
    finally:
        gkd_trainer.gkd_on_policy_fraction = 0.0


def test_on_policy_fraction_zero_never_samples_from_student(gkd_trainer, monkeypatch):
    """Default (0.0) → never uses the student sampler."""
    called = {"student": 0, "teacher": 0}
    monkeypatch.setattr(
        gkd_trainer,
        "_sample_from_student",
        lambda u: (
            called.__setitem__("student", called["student"] + 1),
            "student-sample",
        )[1],
    )
    monkeypatch.setattr(
        gkd_trainer.teacher_backend,
        "generate",
        lambda msgs, max_new_tokens=256: (
            called.__setitem__("teacher", called["teacher"] + 1),
            "teacher-sample",
        )[1],
    )
    for _ in range(5):
        out = gkd_trainer._generate_trajectory("q?")
        assert out == "teacher-sample"
    assert called["student"] == 0
    assert called["teacher"] == 5


def test_local_toggle_path_unchanged():
    """No teacher_backend → falls back to adapter-toggle KL, dense path."""
    tokenizer = _mk_tokenizer()
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model = get_peft_model(
        model,
        PeftLoraConfig(
            r=4, lora_alpha=8, target_modules=["c_attn"], task_type="CAUSAL_LM"
        ),
    )
    trainer = ContextBakingTrainer(
        model=model,
        args=BakeryConfig(
            output_dir="/tmp/bakery_local_toggle",
            num_trajectories=1,
            trajectory_length=8,
            per_device_train_batch_size=1,
            num_train_epochs=1,
            logging_steps=1,
            report_to="none",
            use_cpu=True,
        ),
        context_config=ContextConfig(
            prefix_messages=[{"role": "system", "content": "s"}]
        ),
        processing_class=tokenizer,
        data_collator=prompt_baking_collator,
    )
    clear_mask_cache()
    payload = trainer._teacher_forward(
        trainer.model,
        trainer._build_batch(
            {"user_messages": ["q"], "responses": ["a"]}, trainer.model
        ),
    )
    assert payload[0] == "dense"
