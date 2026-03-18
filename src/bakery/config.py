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
        metadata={"help": "System prompt text to bake into model weights."},
    )
    system_prompt_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to file containing the system prompt."},
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
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        metadata={"help": "Modules to apply LoRA to."},
    )
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})
    bias: str = field(default="none", metadata={"help": "LoRA bias mode."})
