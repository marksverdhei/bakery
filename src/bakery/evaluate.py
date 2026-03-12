"""Evaluation utilities for prompt baking."""

import re

import torch
from typing import List, Tuple, Optional


def evaluate_model(
    model,
    tokenizer,
    qa_pairs: List[Tuple[str, List[str]]],
    desc: str = "Model",
    system_prompt: Optional[str] = None,
) -> dict:
    """Evaluate model on Q&A pairs via keyword matching.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Matching tokenizer.
        qa_pairs: List of (question, expected_keywords) tuples.
        desc: Label for logging.
        system_prompt: Optional system prompt to prepend.

    Returns:
        Dict with accuracy, correct, total, and per-question results.
    """
    if not qa_pairs:
        return {"accuracy": 0, "correct": 0, "total": 0, "results": []}

    was_training = model.training
    model.eval()
    correct = 0
    results = []

    for question, expected_keywords in qa_pairs:
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
        else:
            messages = [{"role": "user", "content": question}]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
            model.device
        )

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        response_clean = (
            re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip().lower()
        )

        is_correct = any(kw in response_clean for kw in expected_keywords)
        if is_correct:
            correct += 1

        results.append(
            {
                "question": question,
                "response": response_clean[:200],
                "correct": is_correct,
            }
        )

    if was_training:
        model.train()

    accuracy = correct / len(qa_pairs)
    print(f"  {desc}: {correct}/{len(qa_pairs)} ({accuracy * 100:.1f}%)")
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(qa_pairs),
        "results": results,
    }
