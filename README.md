# bakery

*Where LLMs go to get baked.*

Bakery distills arbitrary **prefix contexts** — system prompts, few-shot examples, conversation histories, accumulated memories — into model weights via KL-divergence training with LoRA. You get the behavior of a context-conditioned model at zero inference-time prompt cost.

Based on [Prompt Baking](https://arxiv.org/abs/2409.13697), generalized to arbitrary prefix contexts.

## How it works

A single model serves as both teacher and student through PEFT adapter toggling:

- **Teacher** (adapters disabled): sees the full prefix context, generates reference behavior
- **Student** (adapters enabled): sees no prefix (or only the last N messages of it), trained to match the teacher's output distribution

The training objective minimizes per-token KL divergence between teacher and student logits on whichever message tokens you mark as targets (by default: all assistant turns).

## Data sources

The `dataset` field accepts a local JSON file or a HuggingFace dataset ID (auto-detected). The format determines the training mode:

| Data format | Training mode |
|-------------|--------------|
| Prompts only (list of strings, prompt-only columns) | On-the-fly trajectory generation from teacher |
| Paired data (prompt+response, chat messages) | Train directly on precomputed pairs |

You can also use `training_prompts` for inline prompt lists in YAML.

## Install

```bash
pip install git+https://github.com/marksverdhei/bakery.git
```

Or for development:
```bash
git clone https://github.com/marksverdhei/bakery.git
cd bakery
uv sync --dev
uv pip install -e .
```

## Quick start

```bash
bakery --config examples/basic.yaml
```

All config is flat YAML parsed by `HfArgumentParser`, so any `TrainingArguments` field works:

```yaml
# Standard HF training
output_dir: "./outputs/my_bake"
num_train_epochs: 3
learning_rate: 1e-4
bf16: true

# Prompt baking
system_prompt: "You are a helpful assistant."
num_trajectories: 4
trajectory_length: 128

# Model
model_name_or_path: "Qwen/Qwen3-0.6B"

# LoRA
r: 64
lora_alpha: 128

# Data
training_prompts:
  - "What is the capital of France?"
  - "Explain photosynthesis."
```

Override any field from CLI:
```bash
bakery --config examples/basic.yaml --num_train_epochs 5 --learning_rate 5e-5
```

## As a library

```python
from bakery import (
    BakeryConfig,
    ContextConfig,
    ContextBakingTrainer,
    create_dataset,
    prompt_baking_collator,
)

config = BakeryConfig(
    output_dir="./outputs",
    num_train_epochs=3,
    learning_rate=1e-4,
)

context_config = ContextConfig(
    prefix_messages=[
        {"role": "system", "content": "You are helpful."},
    ],
)

dataset = create_dataset(
    ["What is AI?", "Explain gravity."],
    ["AI is...", "Gravity is..."],  # optional precomputed responses
)

trainer = ContextBakingTrainer(
    model=peft_model,
    args=config,
    context_config=context_config,
    train_dataset=dataset,
    processing_class=tokenizer,
    data_collator=prompt_baking_collator,
)
trainer.train()
```

## Context baking

Beyond a single system prompt, bakery supports arbitrary prefix contexts via `ContextConfig` — flat YAML fields alongside the rest:

```yaml
# Any list of chat messages. Teacher sees these prepended to every example.
prefix_messages:
  - {role: system, content: "You answer concisely."}
  - {role: user, content: "Example Q"}
  - {role: assistant, content: "Example A"}

# How many trailing prefix messages the *student* also sees.
# 0 (default) = pure baking. N>0 = last N messages are kept at inference.
student_retained_turns: 0

# Which message roles contribute tokens to the KL loss.
target_roles: [assistant]

# Optional regex (re.search) over message content to further restrict targets.
# Useful to bake only final answers while ignoring chain-of-thought.
target_content_pattern: "^Answer:"
```

Alternatively, load the prefix from a file:

```yaml
prefix_messages_file: "./prefixes/persona_A.yaml"  # or .json
```

**Per-row prefixes.** A `prefix_messages` column on the dataset overrides the global one, so you can bake many contexts into a single adapter.

**Multi-turn datasets.** HuggingFace chat datasets (`messages` column) are loaded verbatim: each row's full conversation becomes the teacher view, and `student_retained_turns` controls how much of it the student sees.

**Migration from `system_prompt`.** The old `system_prompt: "..."` field still works — it auto-desugars to `prefix_messages: [{role: system, content: ...}]` and emits a `DeprecationWarning`. Prefer the new field for new configs.

## Examples

| Config | Description |
|--------|-------------|
| [`examples/basic.yaml`](examples/basic.yaml) | On-the-fly trajectory generation from inline prompts |
| [`examples/sft_dataset.yaml`](examples/sft_dataset.yaml) | Bake from an existing HF chat dataset |
| [`examples/multi_turn_prefix.yaml`](examples/multi_turn_prefix.yaml) | System prompt + few-shot demonstration prefix |
| [`examples/continual_memory.yaml`](examples/continual_memory.yaml) | Multi-turn HF chat dataset with `student_retained_turns: 2` |
| [`examples/per_row_prefix.yaml`](examples/per_row_prefix.yaml) | Per-row persona/context prefixes from a JSON dataset |
| [`examples/pattern_targets.yaml`](examples/pattern_targets.yaml) | Regex-filtered KL targets (bake final answers only) |
