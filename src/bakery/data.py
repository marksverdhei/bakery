"""Data loading utilities for prompt baking."""

import json
from typing import List, Optional, Tuple

from datasets import Dataset

from bakery.config import BakeryConfig, DataConfig


def create_dataset(
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


def load_data(
    data_config: Optional[DataConfig] = None,
    *,
    dataset: Optional[str] = None,
    dataset_split: str = "train",
    training_prompts: Optional[List[str]] = None,
) -> Tuple[List[str], Optional[List[str]]]:
    """Load training data from config or explicit parameters.

    Can be called with a DataConfig object (backward-compatible) or with
    explicit keyword arguments for dataset/split/training_prompts.

    Returns:
        (prompts, responses) where responses is None if only prompts are
        available (triggering on-the-fly trajectory generation).
    """
    if data_config is not None:
        dataset = dataset or data_config.dataset
        dataset_split = (
            data_config.dataset_split
            if dataset == data_config.dataset
            else dataset_split
        )
        training_prompts = training_prompts or data_config.training_prompts

    if dataset:
        return load_dataset(dataset, dataset_split)

    if training_prompts:
        return training_prompts, None

    raise ValueError(
        "No training data configured. Provide 'dataset' or 'training_prompts'."
    )


def load_dataset(
    source: str, split: str = "train"
) -> Tuple[List[str], Optional[List[str]]]:
    """Load training data from a local JSON file or HuggingFace dataset.

    Auto-detects:
    - Local file vs HF dataset (by checking if path exists on disk)
    - Paired data (prompt+response) vs prompts-only

    Supported formats:

    Local JSON:
        - ["prompt1", "prompt2", ...]                          → prompts only
        - [{"prompt": ..., "response": ...}, ...]              → paired
        - {"pairs": [{"prompt": ..., "response": ...}]}        → paired
        - [{"question": ...}, ...]  (no response key)          → prompts only

    HuggingFace dataset:
        - Rows with 'messages' column (chat format)            → paired
        - Rows with 'prompt'/'input'/'question' column         → prompts only

    Returns:
        (prompts, responses) where responses is None for prompts-only data.
    """
    import os

    if os.path.exists(source):
        return _load_json(source)
    return _load_hf(source, split)


def _load_json(path: str) -> Tuple[List[str], Optional[List[str]]]:
    with open(path) as f:
        data = json.load(f)

    # Unwrap nested containers
    if isinstance(data, dict):
        for key in (
            "pairs",
            "data",
            "samples",
            "completions",
            "training_samples",
            "training_prompts",
            "prompts",
            "questions",
        ):
            if key in data:
                data = data[key]
                break

    # Plain list of strings → prompts only
    if isinstance(data, list) and data and isinstance(data[0], str):
        return data, None

    # List of dicts → check if responses are present
    prompts, responses = [], []
    has_responses = False
    for item in data:
        p = item.get(
            "prompt",
            item.get("user_message", item.get("input", item.get("question", ""))),
        )
        r = item.get("response", item.get("completion", item.get("output", "")))
        if p:
            prompts.append(p)
            responses.append(r)
            if r:
                has_responses = True

    if has_responses:
        return prompts, responses
    return prompts, None


def _load_hf(
    dataset_id: str, split: str = "train"
) -> Tuple[List[str], Optional[List[str]]]:
    from datasets import load_dataset as hf_load_dataset

    ds = hf_load_dataset(dataset_id, split=split)
    columns = ds.column_names

    # Chat format with messages column → extract (user, assistant) pairs
    if "messages" in columns:
        prompts, responses = [], []
        for row in ds:
            messages = row["messages"]
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

    # Find prompt column
    prompt_col = None
    for col in ("prompt", "input", "question", "text", "instruction"):
        if col in columns:
            prompt_col = col
            break

    if prompt_col is None:
        raise ValueError(
            f"Cannot find prompt column in dataset '{dataset_id}'. "
            f"Available columns: {columns}"
        )

    # Check for response column
    response_col = None
    for col in ("response", "completion", "output", "answer", "target"):
        if col in columns:
            response_col = col
            break

    prompts, responses = [], []
    for row in ds:
        if row[prompt_col]:
            prompts.append(row[prompt_col])
            if response_col:
                responses.append(row[response_col])

    if response_col:
        return prompts, responses

    return prompts, None


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
        keywords = (
            [answer.lower()] if isinstance(answer, str) else [a.lower() for a in answer]
        )
        qa_pairs.append((question, keywords))

    return qa_pairs
