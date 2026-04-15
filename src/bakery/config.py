"""Configuration dataclasses for prompt baking.

BakeryConfig extends TrainingArguments so all standard HF training fields
are available. Only prompt-baking-specific fields are added here.
The entire config is flat YAML parseable via HfArgumentParser.
"""

from dataclasses import dataclass, field
from typing import Optional, List

from transformers import TrainingArguments


@dataclass
class BakeryConfig(TrainingArguments):
    """Training arguments for prompt baking via KL divergence.

    Extends TrainingArguments with prompt-baking-specific fields.
    All standard fields (learning_rate, num_train_epochs, per_device_train_batch_size,
    max_grad_norm, optim, lr_scheduler_type, warmup_ratio, logging_steps,
    save_strategy, output_dir, gradient_checkpointing, bf16, seed, etc.) are inherited.
    """

    system_prompt: Optional[str] = field(
        default=None,
        metadata={
            "help": "DEPRECATED — use ContextConfig.prefix_messages instead. "
            "When set, desugars to prefix_messages=[{role: system, content: ...}]."
        },
    )
    system_prompt_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "DEPRECATED — use ContextConfig.prefix_messages_file instead. "
            "Path to file containing the system prompt."
        },
    )
    num_trajectories: int = field(
        default=4,
        metadata={"help": "Number of trajectory samples per prompt per step."},
    )
    trajectory_length: int = field(
        default=128,
        metadata={"help": "Maximum new tokens per generated trajectory."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for KL divergence softmax scaling."},
    )
    sampling_temperature: float = field(
        default=0.8,
        metadata={"help": "Temperature for trajectory generation sampling."},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "Truncate sequences to this many tokens. None = no truncation."
        },
    )
    sequential_eval: bool = field(
        default=False,
        metadata={
            "help": "Run teacher and student forward passes sequentially during eval, "
            "offloading teacher logits to CPU between passes. Reduces peak VRAM at the "
            "cost of extra CPU-GPU transfers. Useful for large models that OOM with "
            "both passes in memory simultaneously."
        },
    )

    def __post_init__(self):
        self.remove_unused_columns = False

        # HfArgumentParser.parse_yaml_file sometimes passes numeric values as strings
        for attr in (
            "learning_rate",
            "temperature",
            "sampling_temperature",
            "warmup_ratio",
            "max_grad_norm",
            "weight_decay",
        ):
            val = getattr(self, attr, None)
            if isinstance(val, str):
                try:
                    setattr(self, attr, float(val))
                except ValueError:
                    raise ValueError(f"Cannot convert {attr}={val!r} to float")
        for attr in (
            "num_trajectories",
            "trajectory_length",
            "num_train_epochs",
            "logging_steps",
            "seed",
        ):
            val = getattr(self, attr, None)
            if isinstance(val, str):
                try:
                    setattr(self, attr, int(val))
                except ValueError:
                    raise ValueError(f"Cannot convert {attr}={val!r} to int")

        super().__post_init__()

        if self.system_prompt is None and self.system_prompt_file is not None:
            with open(self.system_prompt_file) as f:
                self.system_prompt = f.read().strip()


@dataclass
class DataConfig:
    """Data source and model loading configuration."""

    model_name_or_path: str = field(
        default="Qwen/Qwen3-0.6B",
        metadata={"help": "HuggingFace model name or path."},
    )
    torch_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Model dtype: float32, float16, bfloat16."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Trust remote code when loading model."},
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Load model in 4-bit quantization (QLoRA)."},
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "4-bit quantization type: nf4 or fp4."},
    )
    bnb_4bit_use_double_quant: bool = field(
        default=True,
        metadata={"help": "Use double quantization for 4-bit."},
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": "Attention implementation: flash_attention_2, sdpa, eager, or None for default."
        },
    )
    auto_install_optional_deps: bool = field(
        default=True,
        metadata={
            "help": "Auto-install missing optional dependencies at runtime. "
            "Disabled when HF_HUB_OFFLINE=1 is set."
        },
    )
    use_unsloth: bool = field(
        default=False,
        metadata={
            "help": "Use Unsloth for optimized model loading and LoRA training. "
            "Provides ~2x speedup and 60-70%% VRAM reduction."
        },
    )

    # Training data — provide ONE of these
    dataset: Optional[str] = field(
        default=None,
        metadata={
            "help": "Training data source: a local JSON file or HuggingFace dataset ID. "
            "Auto-detects format. Datasets with responses (messages, prompt+response "
            "pairs) skip trajectory generation. Datasets with only prompts use "
            "on-the-fly generation from the teacher."
        },
    )
    dataset_split: str = field(
        default="train",
        metadata={"help": "Dataset split to use (only for HuggingFace datasets)."},
    )
    training_prompts: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Inline list of training prompts (on-the-fly generation)."},
    )

    eval_dataset_split: Optional[str] = field(
        default=None,
        metadata={
            "help": "Dataset split for validation loss (e.g. 'test_sft[:200]'). "
            "Uses the same 'dataset' source. Combine with eval_strategy/eval_steps."
        },
    )

    # Evaluation
    eval_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to JSON with evaluation Q&A pairs."},
    )
    heldout_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to JSON with held-out test Q&A pairs."},
    )

    # Knowledge baking (corpus-based system prompt)
    corpus_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to JSON corpus file (for knowledge baking)."},
    )
    corpus_format: str = field(
        default="papers",
        metadata={"help": "Corpus format: papers, text, list, custom."},
    )
    system_prompt_template: Optional[str] = field(
        default=None,
        metadata={"help": "Template with {corpus} placeholder."},
    )


@dataclass
class LoraConfig:
    """LoRA adapter configuration."""

    r: int = field(default=64, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=128, metadata={"help": "LoRA alpha scaling."})
    target_modules: Optional[List[str]] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        metadata={
            "help": "Modules to apply LoRA to. Use 'all-linear' or 'all' to target all linear layers."
        },
    )

    def __post_init__(self):
        # Support "all" as shorthand for "all-linear" (PEFT convention)
        if self.target_modules == "all" or self.target_modules == ["all"]:
            self.target_modules = "all-linear"
        elif self.target_modules == ["all-linear"]:
            self.target_modules = "all-linear"

    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})
    bias: str = field(default="none", metadata={"help": "LoRA bias mode."})


@dataclass
class ContextConfig:
    """Prefix context and target-mask configuration for context baking.

    Generalizes system-prompt baking to arbitrary prefix contexts (conversation
    histories, accumulated memories, few-shot examples). The teacher sees the
    full prefix; the student sees an optionally-trimmed version. KL is computed
    only on tokens matching `target_roles` / `target_content_pattern`.
    """

    prefix_messages: Optional[List[dict]] = field(
        default=None,
        metadata={
            "help": "Global prefix context as a list of {role, content} dicts. "
            "Teacher sees this prepended to every example; student does not "
            "(or sees only the last student_retained_turns of it). Overridden "
            "per-row by a 'prefix_messages' dataset column if present."
        },
    )
    prefix_messages_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a JSON or YAML file containing the prefix_messages list. "
            "Loaded at CLI parse time if prefix_messages is not set inline."
        },
    )
    student_retained_turns: int = field(
        default=0,
        metadata={
            "help": "Number of trailing prefix messages the student also sees. "
            "0 (default) = pure baking: student sees no prefix. N>0 = student "
            "sees the last N messages of the prefix; earlier messages are baked."
        },
    )
    target_roles: List[str] = field(
        default_factory=lambda: ["assistant"],
        metadata={
            "help": "Message roles whose tokens receive KL loss. Default: ['assistant']. "
            "Any message whose role is in this list is treated as a training target."
        },
    )
    target_content_pattern: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional regex applied to message content. When set, a message "
            "becomes a target only if its role is in target_roles AND its content "
            "matches this pattern (re.search semantics)."
        },
    )
