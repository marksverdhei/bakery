"""KL divergence computation for prompt baking."""

import torch
import torch.nn.functional as F
from contextlib import contextmanager


def compute_kl_divergence(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
    per_sample: bool = False,
) -> torch.Tensor:
    """Compute D_KL(P_teacher || P_student) per-token, masked and averaged.

    Args:
        teacher_logits: [batch, seq_len, vocab_size]
        student_logits: [batch, seq_len, vocab_size]
        mask: [batch, seq_len] attention mask (1=real, 0=padding)
        temperature: Softening temperature for distributions
        per_sample: If True, return per-sample averaged losses [batch]
                    instead of a single scalar.

    Returns:
        Scalar KL divergence loss, or [batch] tensor if per_sample=True.
    """
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # F.kl_div(input, target) computes D_KL(target || input), i.e. forward KL
    # from teacher to student — matching the prompt baking paper (2409.13697).
    kl_per_token = F.kl_div(
        student_log_probs, teacher_probs, reduction="none", log_target=False
    ).sum(dim=-1)

    masked_kl = kl_per_token * mask

    if per_sample:
        # Per-sample: average each sample's KL over its own unmasked tokens.
        num_tokens = mask.sum(dim=-1)  # [batch]
        safe_num = num_tokens.clamp(min=1.0)
        return masked_kl.sum(dim=-1) / safe_num  # [batch]

    num_tokens = mask.sum()
    return masked_kl.sum() / num_tokens if num_tokens > 0 else masked_kl.sum()


def topk_forward_kl(
    student_logits: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
    per_sample: bool = False,
) -> torch.Tensor:
    """Forward KL over the teacher's top-k support.

    Works for both dense (K = vocab) and sparse (K = 20) teachers. The teacher's
    top-k logprobs are assumed to be renormalized over those K (so exp(t).sum=1
    along the last dim) — this is what `topk_from_logits` produces and what we
    require API backends to deliver.

    KL is computed as: sum_k p_teacher * (log p_teacher - log p_student_at_topk),
    where p_student_at_topk is the student's softmax restricted (via gather) to
    the teacher's top-k indices and re-normalized over those K so the comparison
    is on the same support.

    Args:
        student_logits:        (B, T, V) student logits over full vocab.
        teacher_topk_indices:  (B, T, K) vocab ids of teacher's top-k.
        teacher_topk_logprobs: (B, T, K) renormalized log-probs over those K.
        mask:                  (B, T) target mask (1 where loss applies).
        temperature:           softmax temperature applied to student logits.
        per_sample:            if True, return [B] averaged per-sample.

    Returns:
        Scalar KL, or [B] if per_sample=True.
    """
    student_logprobs_full = F.log_softmax(student_logits / temperature, dim=-1)
    student_logprobs_at_topk = torch.gather(
        student_logprobs_full, dim=-1, index=teacher_topk_indices
    )  # (B, T, K)
    # Renormalize student over the same top-k support so we compare apples to
    # apples (matches the teacher's renormalization).
    student_logprobs_renorm = (
        student_logprobs_at_topk
        - student_logprobs_at_topk.logsumexp(dim=-1, keepdim=True)
    )

    p_teacher = teacher_topk_logprobs.exp()
    kl_per_pos = (p_teacher * (teacher_topk_logprobs - student_logprobs_renorm)).sum(
        dim=-1
    )  # (B, T)

    mask_f = mask.float()
    masked_kl = kl_per_pos * mask_f
    if per_sample:
        num_tokens = mask_f.sum(dim=-1).clamp(min=1.0)
        return masked_kl.sum(dim=-1) / num_tokens
    num_tokens = mask_f.sum()
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
