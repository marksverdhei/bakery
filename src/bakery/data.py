"""Data loading utilities for context baking."""

import json
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset

from bakery.config import BakeryConfig, DataConfig


def create_dataset(
    prompts: List[str],
    responses: Optional[List[str]] = None,
) -> Dataset:
    """Create a HuggingFace Dataset for single-turn context baking training.

    Args:
        prompts: List of user messages.
        responses: Optional precomputed responses. When provided, the trainer
            skips on-the-fly trajectory generation.
    """
    data = {"user_messages": prompts}
    if responses is not None:
        data["responses"] = responses
    return Dataset.from_dict(data)


def create_conversational_dataset(
    rows: List[Dict[str, Any]],
) -> Dataset:
    """Create a HuggingFace Dataset from conversational rows.

    Each row is a dict with:
      - `turns`: List[{role, content}] — the training conversation (required).
      - `prefix_messages`: Optional[List[{role, content}]] — per-row prefix
        that overrides the global ContextConfig.prefix_messages.
      - `response`: Optional[str] — for trajectory mode or single-turn data;
        when set, the trainer appends {role: assistant, content: response}
        to `turns`.
    """
    data = {
        "turns": [r.get("turns", []) for r in rows],
        "prefix_messages": [r.get("prefix_messages") for r in rows],
        "responses": [r.get("response") for r in rows],
    }
    return Dataset.from_dict(data)


def prompt_baking_collator(features: List[dict]) -> dict:
    """Collate function for context baking.

    Accepts both shapes and emits a batch with whichever keys are present:

    - Legacy: {user_messages, responses}
    - Conversational: {prefix_messages, turns, responses}

    Legacy items are passed through unchanged so the trainer can normalize
    them to the conversational shape.
    """
    user_messages: List[str] = []
    responses: List[Optional[str]] = []
    turns: List[List[dict]] = []
    prefix_messages: List[Optional[List[dict]]] = []
    saw_turns = False

    for f in features:
        if "turns" in f:
            saw_turns = True
            turns.append(list(f["turns"]) if f["turns"] is not None else [])
            prefix_messages.append(f.get("prefix_messages"))
            responses.append(f.get("responses"))
            continue

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

    if saw_turns:
        return {
            "turns": turns,
            "prefix_messages": prefix_messages,
            "responses": responses,
        }
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
    data_config: DataConfig,
) -> Tuple[List[str], Optional[List[str]]]:
    """Load training data from config.

    Returns:
        (prompts, responses) where responses is None if only prompts are
        available (triggering on-the-fly trajectory generation).
    """
    if data_config.dataset:
        return load_dataset(data_config.dataset, data_config.dataset_split)

    if data_config.training_prompts:
        return data_config.training_prompts, None

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


def load_conversations(
    source: str, split: str = "train"
) -> List[Dict[str, Any]]:
    """Load conversational training rows from a local JSON file or HF dataset.

    Returns a list of dicts suitable for `create_conversational_dataset`:
      - `turns`: List[{role, content}]
      - `prefix_messages`: Optional[List[{role, content}]]
      - `response`: Optional[str]

    HF `messages` columns are preserved as full turn sequences (NOT flattened),
    so multi-turn conversations can be baked with all assistant turns as KL targets.
    """
    import os

    if os.path.exists(source):
        return _load_conversations_json(source)
    return _load_conversations_hf(source, split)


def _load_conversations_json(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        for key in (
            "pairs",
            "data",
            "samples",
            "completions",
            "training_samples",
            "conversations",
        ):
            if key in data:
                data = data[key]
                break

    if isinstance(data, list) and data and isinstance(data[0], str):
        return [
            {"turns": [{"role": "user", "content": p}], "prefix_messages": None, "response": None}
            for p in data
        ]

    rows: List[Dict[str, Any]] = []
    for item in data:
        prefix = item.get("prefix_messages")
        if "messages" in item and isinstance(item["messages"], list):
            turns = item["messages"]
            response = None
        else:
            p = item.get(
                "prompt",
                item.get("user_message", item.get("input", item.get("question", ""))),
            )
            if not p:
                continue
            turns = [{"role": "user", "content": p}]
            response = item.get("response", item.get("completion", item.get("output")))
        rows.append(
            {
                "turns": list(turns),
                "prefix_messages": list(prefix) if prefix else None,
                "response": response,
            }
        )
    return rows


def _load_conversations_hf(
    dataset_id: str, split: str = "train"
) -> List[Dict[str, Any]]:
    from datasets import load_dataset as hf_load_dataset

    ds = hf_load_dataset(dataset_id, split=split)
    columns = ds.column_names
    has_prefix = "prefix_messages" in columns

    if "messages" in columns:
        rows: List[Dict[str, Any]] = []
        for row in ds:
            msgs = row["messages"]
            if not msgs:
                continue
            rows.append(
                {
                    "turns": list(msgs),
                    "prefix_messages": list(row["prefix_messages"])
                    if has_prefix and row.get("prefix_messages")
                    else None,
                    "response": None,
                }
            )
        return rows

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

    response_col = None
    for col in ("response", "completion", "output", "answer", "target"):
        if col in columns:
            response_col = col
            break

    rows = []
    for row in ds:
        if not row.get(prompt_col):
            continue
        rows.append(
            {
                "turns": [{"role": "user", "content": row[prompt_col]}],
                "prefix_messages": list(row["prefix_messages"])
                if has_prefix and row.get("prefix_messages")
                else None,
                "response": row[response_col] if response_col else None,
            }
        )
    return rows


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
