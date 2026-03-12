# bakery

*Where LLMs go to get baked.*

Prompt baking distills a system prompt into model weights via KL divergence training with LoRA, so you get the behavior of a prompted model at zero inference-time prompt cost.

Based on [Prompt Baking](https://arxiv.org/abs/2409.13697).

## How it works

A single model serves as both teacher and student through PEFT adapter toggling:

- **Teacher** (adapters disabled): sees the system prompt, generates reference behavior
- **Student** (adapters enabled): no system prompt, trained to match the teacher's output distribution

The training objective minimizes per-token KL divergence between teacher and student logits on the response portion of each conversation.

## Data sources

Bakery supports two modes for training data:

| Mode | Config field | Description |
|------|-------------|-------------|
| **On-the-fly** | `training_prompts` / `training_prompts_file` | Teacher generates trajectories during training |
| **Precomputed** | `dataset` | Provide (prompt, response) pairs from a local JSON file or HF dataset (auto-detected) |

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
from bakery import BakeryConfig, PromptBakingTrainer, PromptBakingDataset, prompt_baking_collator

config = BakeryConfig(
    output_dir="./outputs",
    system_prompt="You are helpful.",
    num_train_epochs=3,
    learning_rate=1e-4,
)

dataset = PromptBakingDataset(
    prompts=["What is AI?", "Explain gravity."],
    responses=["AI is...", "Gravity is..."],  # optional precomputed
)

trainer = PromptBakingTrainer(
    model=peft_model,
    args=config,
    train_dataset=dataset,
    processing_class=tokenizer,
    data_collator=prompt_baking_collator,
)
trainer.train()
```

## Examples

| Config | Description |
|--------|-------------|
| [`examples/basic.yaml`](examples/basic.yaml) | On-the-fly trajectory generation from inline prompts |
| [`examples/sft_dataset.yaml`](examples/sft_dataset.yaml) | Bake from an existing HF chat dataset |
