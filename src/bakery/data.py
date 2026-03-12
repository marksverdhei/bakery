"""Data loading utilities for prompt baking."""

import json
from typing import List, Optional, Tuple

from datasets import Dataset

from bakery.config import BakeryConfig, DataConfig


def PromptBakingDataset(
    prompts: List[str],
    responses: Optional[List[str]] = None,
) -> Dataset:
    """Create a HuggingFace Dataset for prompt baking training.

    Args:
        prompts: List of user messages.
        responses: Optional precomputed responses. When provided, the trainer
            skips on-the-fly trajectory generation.
    """
    data = {"user_messages": prompts}
    if responses is not None:
        data["responses"] = responses
    return Dataset.from_dict(data)


def prompt_baking_collator(features: List[dict]) -> dict:
    """Collate function that passes user_messages and responses through."""
    user_messages, responses = [], []
    for f in features:
        msg = f.get("user_messages")
        if isinstance(msg, list):
            user_messages.extend(msg)
        elif isinstance(msg, str):
            user_messages.append(msg)

        resp = f.get("responses")
        if isinstance(resp, list):
            responses.extend(resp)
        elif isinstance(resp, str):
            responses.append(resp)
    return {"user_messages": user_messages, "responses": responses}


def load_corpus(data_config: DataConfig) -> Optional[str]:
    """Load knowledge corpus from file."""
    if not data_config.corpus_file:
        return None

    with open(data_config.corpus_file) as f:
        data = json.load(f)

    if data_config.corpus_format == "papers":
        parts = []
        for i, paper in enumerate(data):
            parts.append(f"Paper {i + 1}: {paper['title']}\n{paper['abstract']}")
        return "\n\n".join(parts)
    elif data_config.corpus_format == "text":
        return data if isinstance(data, str) else data.get("text", "")
    elif data_config.corpus_format == "list":
        return "\n\n".join(data)
    else:
        return json.dumps(data, indent=2)


def build_system_prompt(
    baking_config: BakeryConfig,
    data_config: DataConfig,
    corpus: Optional[str] = None,
) -> str:
    """Build the system prompt from config."""
    if baking_config.system_prompt:
        return baking_config.system_prompt

    if data_config.system_prompt_template and corpus:
        return data_config.system_prompt_template.format(corpus=corpus)

    if corpus:
        return (
            "You are an AI assistant with specialized knowledge. "
            "Here is your knowledge base:\n\n"
            f"{corpus}\n\n"
            "Answer questions about this knowledge accurately and concisely."
        )

    raise ValueError(
        "No system prompt configured. Provide system_prompt, system_prompt_file, "
        "or corpus_file."
    )


def load_training_prompts(data_config: DataConfig) -> List[str]:
    """Load training prompts from config."""
    if data_config.training_prompts:
        return data_config.training_prompts

    if data_config.training_prompts_file:
        with open(data_config.training_prompts_file) as f:
            data = json.load(f)

        if isinstance(data, list):
            if isinstance(data[0], str):
                return data
            elif isinstance(data[0], dict):
                return [
                    item.get("question", item.get("input", item.get("prompt", "")))
                    for item in data
                ]

        if isinstance(data, dict):
            for key in ["training_samples", "training_prompts", "prompts", "questions"]:
                if key in data:
                    items = data[key]
                    if isinstance(items[0], str):
                        return items
                    return [
                        item.get("question", item.get("input", "")) for item in items
                    ]

    raise ValueError(
        "No training prompts configured. "
        "Provide training_prompts or training_prompts_file."
    )


def load_sft_dataset(
    dataset_id: str, split: str = "train"
) -> Tuple[List[str], List[str]]:
    """Load (user, assistant) pairs from a HuggingFace chat dataset.

    Expects rows with a 'messages' field containing a list of
    {"role": "user"|"assistant", "content": "..."} dicts.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split=split)
    prompts, responses = [], []

    for row in ds:
        messages = row.get("messages", [])
        for i, msg in enumerate(messages):
            if msg["role"] == "assistant" and msg.get("content", "").strip():
                user_msg = ""
                for j in range(i - 1, -1, -1):
                    if messages[j]["role"] == "user":
                        user_msg = messages[j]["content"]
                        break
                if user_msg:
                    prompts.append(user_msg)
                    responses.append(msg["content"])

    return prompts, responses


def load_precomputed_responses(path: str) -> Tuple[List[str], List[str]]:
    """Load precomputed (prompt, response) pairs from JSON file.

    Supported formats:
        - [{"prompt": ..., "response": ...}, ...]
        - {"pairs": [{"prompt": ..., "response": ...}]}
    """
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        for key in ("pairs", "data", "samples", "completions"):
            if key in data:
                data = data[key]
                break

    prompts, responses = [], []
    for item in data:
        p = item.get("prompt", item.get("user_message", item.get("input", "")))
        r = item.get("response", item.get("completion", item.get("output", "")))
        if p and r:
            prompts.append(p)
            responses.append(r)

    return prompts, responses


def load_eval_data(eval_file: Optional[str]) -> List[Tuple[str, List[str]]]:
    """Load evaluation Q&A pairs from file."""
    if not eval_file:
        return []

    with open(eval_file) as f:
        data = json.load(f)

    qa_pairs = []
    items = data
    if isinstance(data, dict):
        for key in ["evaluation_samples", "test_samples", "eval", "qa_pairs"]:
            if key in data:
                items = data[key]
                break

    for item in items:
        question = item.get("question", item.get("input", ""))
        answer = item.get("answer", item.get("expected", item.get("target", "")))
        keywords = [answer.lower()] if isinstance(answer, str) else [a.lower() for a in answer]
        qa_pairs.append((question, keywords))

    return qa_pairs
