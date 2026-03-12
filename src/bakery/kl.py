"""KL divergence computation for prompt baking."""

import torch
import torch.nn.functional as F
from contextlib import contextmanager


def compute_kl_divergence(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute D_KL(P_teacher || P_student) per-token, masked and averaged.

    Args:
        teacher_logits: [batch, seq_len, vocab_size]
        student_logits: [batch, seq_len, vocab_size]
        mask: [batch, seq_len] attention mask (1=real, 0=padding)
        temperature: Softening temperature for distributions

    Returns:
        Scalar KL divergence loss.
    """
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # F.kl_div(input, target) computes D_KL(target || input), i.e. forward KL
    # from teacher to student — matching the prompt baking paper (2409.13697).
    kl_per_token = F.kl_div(
        student_log_probs, teacher_probs, reduction="none", log_target=False
    ).sum(dim=-1)

    masked_kl = kl_per_token * mask
    num_tokens = mask.sum()
    return masked_kl.sum() / num_tokens if num_tokens > 0 else masked_kl.sum()


@contextmanager
def disable_adapters(model):
    """Context manager to temporarily disable LoRA adapters."""
    try:
        model.disable_adapter_layers()
        yield
    finally:
        model.enable_adapter_layers()


@contextmanager
def padding_side(tokenizer, side: str):
    """Context manager to temporarily override tokenizer padding side."""
    original = tokenizer.padding_side
    tokenizer.padding_side = side
    try:
        yield
    finally:
        tokenizer.padding_side = original
