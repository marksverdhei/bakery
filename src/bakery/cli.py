"""Bakery CLI - train prompt-baked models from flat YAML configs.

Usage:
    bakery --config config.yaml
    bakery --config config.yaml --num_train_epochs 5 --learning_rate 1e-4
"""

import argparse
import os
import json
import warnings
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
)
from peft import LoraConfig as PeftLoraConfig, get_peft_model

from bakery.config import BakeryConfig, ContextConfig, DataConfig, LoraConfig
from bakery.trainer import ContextBakingTrainer
from bakery.data import (
    create_conversational_dataset,
    create_dataset,
    prompt_baking_collator,
    load_conversations,
    load_corpus,
    build_system_prompt,
    load_data,
    load_eval_data,
)
from bakery.evaluate import evaluate_model

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _load_prefix_file(path: str) -> list:
    """Load prefix_messages from a JSON or YAML file (expects a list of {role, content})."""
    with open(path) as f:
        text = f.read()
    if path.endswith((".yaml", ".yml")):
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(
            f"prefix_messages_file {path!r} must contain a JSON/YAML list of "
            "{role, content} dicts."
        )
    return data


def main():
    # Pre-parse --config and collect remaining args for HfArgumentParser.
    # argparse handles -h/--help and validates --config before we proceed.
    pre_parser = argparse.ArgumentParser(
        prog="bakery",
        description="Bakery - prompt baking via KL divergence distillation with LoRA.",
        epilog=(
            "All TrainingArguments fields can be passed as CLI overrides on top of "
            "the YAML config. See examples/ for sample configs."
        ),
    )
    pre_parser.add_argument(
        "--config",
        required=True,
        metavar="FILE",
        help="path to YAML config file",
    )
    pre_args, remaining_args = pre_parser.parse_known_args()
    config_file = pre_args.config

    parser = HfArgumentParser((BakeryConfig, DataConfig, LoraConfig, ContextConfig))

    baking_config, data_config, lora_config, context_config = parser.parse_yaml_file(
        config_file, allow_extra_keys=True
    )
    # Apply CLI overrides on top of YAML config.
    # We parse the remaining CLI args into fresh dataclasses, then detect
    # which fields were explicitly set by comparing against a baseline
    # (dataclasses parsed with no args at all).
    if remaining_args:
        # Collect which field names were explicitly passed on the CLI.
        explicit_keys = set()
        for arg in remaining_args:
            if arg.startswith("--"):
                explicit_keys.add(arg.lstrip("-").replace("-", "_"))

        override_parser = HfArgumentParser(
            (BakeryConfig, DataConfig, LoraConfig, ContextConfig)
        )
        overrides = override_parser.parse_args_into_dataclasses(
            args=["--output_dir", baking_config.output_dir] + remaining_args,
            return_remaining_strings=True,
        )
        for override_cfg, base_cfg in zip(
            overrides[:4],
            (baking_config, data_config, lora_config, context_config),
        ):
            for k, v in vars(override_cfg).items():
                if k in explicit_keys:
                    setattr(base_cfg, k, v)

    # Load prefix_messages_file if set (JSON or YAML list of {role, content} dicts).
    if context_config.prefix_messages is None and context_config.prefix_messages_file:
        context_config.prefix_messages = _load_prefix_file(
            context_config.prefix_messages_file
        )

    # Build system prompt (supports corpus-based knowledge baking) — retained
    # for backward compat; desugars into prefix_messages below.
    corpus = load_corpus(data_config)
    try:
        system_prompt = build_system_prompt(baking_config, data_config, corpus)
    except ValueError:
        # No system_prompt configured — OK if prefix_messages is set.
        system_prompt = None
    if system_prompt:
        baking_config.system_prompt = system_prompt

    # Desugar deprecated system_prompt → prefix_messages when user didn't set prefix.
    if not context_config.prefix_messages and baking_config.system_prompt:
        if baking_config.system_prompt_file or corpus:
            pass  # typical (non-deprecated) path: corpus-driven or file-loaded prompt
        else:
            warnings.warn(
                "`system_prompt` is deprecated; use ContextConfig.prefix_messages instead. "
                "Auto-wrapping into prefix_messages=[{role: system, content: ...}].",
                DeprecationWarning,
                stacklevel=2,
            )
        context_config.prefix_messages = [
            {"role": "system", "content": baking_config.system_prompt}
        ]

    if not context_config.prefix_messages:
        raise ValueError(
            "No prefix context configured. Set prefix_messages (inline or via "
            "prefix_messages_file), or the deprecated system_prompt / corpus_file."
        )

    # Load data. Use conversational loader when the source preserves multi-turn
    # history (HF `messages` column or JSON rows with `messages`/`prefix_messages`);
    # otherwise use the simple (prompts, responses) path.
    conversational_rows = None
    training_prompts, precomputed_responses = [], None
    if data_config.dataset or data_config.training_prompts:
        if data_config.dataset:
            try:
                conversational_rows = load_conversations(
                    data_config.dataset, data_config.dataset_split
                )
            except Exception:
                conversational_rows = None
        if conversational_rows is None:
            training_prompts, precomputed_responses = load_data(data_config)
    eval_qa = load_eval_data(data_config.eval_file)
    heldout_qa = load_eval_data(data_config.heldout_file)

    print("=" * 70)
    print("Bakery - Context Baking with KL Divergence")
    print("=" * 70)
    print(
        f"  Prefix messages: {len(context_config.prefix_messages)} "
        f"({sum(len(m.get('content', '')) for m in context_config.prefix_messages):,} chars)"
    )
    if conversational_rows is not None:
        n_turns = sum(len(r.get("turns", [])) for r in conversational_rows)
        print(
            f"  Training rows (conversational): {len(conversational_rows)} "
            f"({n_turns} total turns)"
        )
    else:
        print(f"  Training prompts: {len(training_prompts)}")
        if precomputed_responses:
            print(f"  Precomputed responses: {len(precomputed_responses)}")
        else:
            print("  Mode: on-the-fly trajectory generation")
    if eval_qa:
        print(f"  Evaluation Q&A: {len(eval_qa)}")
    if heldout_qa:
        print(f"  Held-out Q&A: {len(heldout_qa)}")

    # Auto-install optional dependencies based on model and features
    if data_config.auto_install_optional_deps:
        from transformers import AutoConfig
        from bakery.deps import ensure_deps

        model_config = AutoConfig.from_pretrained(
            data_config.model_name_or_path,
            trust_remote_code=data_config.trust_remote_code,
        )
        model_type = getattr(model_config, "model_type", None)
        features = []
        if data_config.load_in_4bit:
            features.append("qlora")
        if data_config.use_unsloth:
            features.append("unsloth")
        ensure_deps(model_type=model_type, features=features)

    # Gemma 4 introduces Gemma4ClippableLinear (inherits nn.Module, not nn.Linear)
    # in its vision/audio encoders.  PEFT rejects it even when targeting only text
    # layers.  Monkey-patch it to inherit nn.Linear so PEFT's type check passes.
    # Must happen before from_pretrained() materialises the model.
    try:
        from transformers.models.gemma4 import modeling_gemma4
        import torch.nn as nn

        class _PatchedClippableLinear(nn.Linear):
            def __init__(self, config, in_features, out_features):
                nn.Linear.__init__(self, in_features, out_features, bias=False)
                self.use_clipped_linears = getattr(config, "use_clipped_linears", False)
                if self.use_clipped_linears:
                    self.register_buffer("input_min", torch.tensor(-float("inf")))
                    self.register_buffer("input_max", torch.tensor(float("inf")))
                    self.register_buffer("output_min", torch.tensor(-float("inf")))
                    self.register_buffer("output_max", torch.tensor(float("inf")))

            def forward(self, x):
                if self.use_clipped_linears:
                    x = torch.clamp(x, self.input_min, self.input_max)
                out = nn.Linear.forward(self, x)
                if self.use_clipped_linears:
                    out = torch.clamp(out, self.output_min, self.output_max)
                return out

        modeling_gemma4.Gemma4ClippableLinear = _PatchedClippableLinear
        print("  Patched Gemma4ClippableLinear for PEFT compatibility")
    except (ImportError, AttributeError):
        pass  # Not a Gemma 4 run or transformers too old

    # Load model
    print(f"\n[1] Loading model: {data_config.model_name_or_path}")
    torch_dtype = DTYPE_MAP.get(data_config.torch_dtype, torch.bfloat16)

    if data_config.use_unsloth:
        from unsloth import FastLanguageModel
        from transformers import AutoConfig as _AC

        _model_cfg = _AC.from_pretrained(
            data_config.model_name_or_path,
            trust_remote_code=data_config.trust_remote_code,
        )
        unsloth_max_seq = baking_config.max_seq_length or getattr(
            _model_cfg, "max_position_embeddings", 4096
        )

        print("  Using Unsloth optimized loading")
        base_model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=data_config.model_name_or_path,
            max_seq_length=unsloth_max_seq,
            dtype=torch_dtype,
            load_in_4bit=data_config.load_in_4bit,
        )
        if data_config.load_in_4bit:
            print("  Loading in 4-bit (QLoRA mode via Unsloth)")
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            data_config.model_name_or_path,
            trust_remote_code=data_config.trust_remote_code,
        )
        load_kwargs = dict(
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=data_config.trust_remote_code,
            low_cpu_mem_usage=True,
        )
        if data_config.attn_implementation:
            load_kwargs["attn_implementation"] = data_config.attn_implementation
        if data_config.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=data_config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=data_config.bnb_4bit_use_double_quant,
            )
            load_kwargs["quantization_config"] = bnb_config
            print("  Loading in 4-bit (QLoRA mode)")
            # Workaround: transformers >=5.x core_model_loading materializes tensors
            # on GPU at full precision before quantization, causing OOM for large models.
            # See: https://github.com/huggingface/transformers/issues/43032
            try:
                base_model = AutoModelForCausalLM.from_pretrained(
                    data_config.model_name_or_path, **load_kwargs
                )
            except torch.OutOfMemoryError:
                print(
                    "  GPU OOM during quantized loading — retrying via CPU + dispatch"
                )
                torch.cuda.empty_cache()
                cpu_kwargs = {**load_kwargs, "device_map": "cpu"}
                base_model = AutoModelForCausalLM.from_pretrained(
                    data_config.model_name_or_path, **cpu_kwargs
                )
                from accelerate import dispatch_model, infer_auto_device_map

                gpu_mem = torch.cuda.get_device_properties(0).total_memory
                device_map = infer_auto_device_map(
                    base_model,
                    max_memory={
                        0: f"{gpu_mem // (1024**3) - 2}GiB",
                        "cpu": "32GiB",
                    },
                )
                base_model = dispatch_model(base_model, device_map=device_map)
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                data_config.model_name_or_path, **load_kwargs
            )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Rough prefix-token count (for summary output only).
    prefix_rendered = tokenizer.apply_chat_template(
        context_config.prefix_messages, tokenize=False, add_generation_prompt=False
    )
    prompt_tokens = len(tokenizer.encode(prefix_rendered, add_special_tokens=False))
    print(f"  Prefix tokens: {prompt_tokens:,}")

    # Baseline evaluations (only when prefix is a single system message; otherwise
    # "discrete prompt" baseline comparison isn't well-defined).
    _is_simple_system_prompt = (
        len(context_config.prefix_messages) == 1
        and context_config.prefix_messages[0].get("role") == "system"
    )
    discrete_sp = (
        context_config.prefix_messages[0]["content"] if _is_simple_system_prompt else None
    )
    if eval_qa:
        print("\n[2] Baseline evaluations...")
        print("\n  === No prefix ===")
        baseline_none = evaluate_model(base_model, tokenizer, eval_qa, "Baseline")
        baseline_discrete = heldout_discrete = None
        if discrete_sp:
            print("\n  === With system prompt (discrete) ===")
            baseline_discrete = evaluate_model(
                base_model,
                tokenizer,
                eval_qa,
                "Discrete prompt",
                system_prompt=discrete_sp,
            )
            if heldout_qa:
                print("\n  === Held-out (discrete prompt) ===")
                heldout_discrete = evaluate_model(
                    base_model,
                    tokenizer,
                    heldout_qa,
                    "Discrete (held-out)",
                    system_prompt=discrete_sp,
                )
    else:
        baseline_none = baseline_discrete = heldout_discrete = None

    # Add LoRA adapters
    print("\n[3] Adding LoRA adapters...")
    if data_config.use_unsloth:
        from unsloth import FastLanguageModel

        model = FastLanguageModel.get_peft_model(
            base_model,
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            target_modules=lora_config.target_modules,
            lora_dropout=0,  # Unsloth optimized kernels require dropout=0
            bias=lora_config.bias,
            use_gradient_checkpointing="unsloth",
            random_state=baking_config.seed,
        )
    else:
        peft_config = PeftLoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            target_modules=lora_config.target_modules,
            lora_dropout=lora_config.lora_dropout,
            bias=lora_config.bias,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, peft_config)
        if data_config.load_in_4bit:
            model.enable_input_require_grads()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # Create dataset
    print("\n[4] Training with KL divergence...")
    if conversational_rows is not None:
        train_dataset = create_conversational_dataset(conversational_rows)
        print(f"  Using {len(train_dataset)} conversational rows")
    else:
        if precomputed_responses:
            print(
                f"  Using {len(precomputed_responses)} precomputed (prompt, response) pairs"
            )
        else:
            print(
                f"  Generating {baking_config.num_trajectories} trajectories per prompt on-the-fly"
            )
        train_dataset = create_dataset(training_prompts, precomputed_responses)

    eval_dataset = None
    if data_config.eval_dataset_split and data_config.dataset:
        print(f"  Loading eval split: {data_config.eval_dataset_split}")
        try:
            eval_rows = load_conversations(
                data_config.dataset, data_config.eval_dataset_split
            )
            eval_dataset = create_conversational_dataset(eval_rows)
        except Exception:
            eval_prompts, eval_responses = load_data(
                type(
                    "_DC",
                    (),
                    {
                        "dataset": data_config.dataset,
                        "dataset_split": data_config.eval_dataset_split,
                        "training_prompts": None,
                    },
                )()
            )
            eval_dataset = create_dataset(eval_prompts, eval_responses)
        print(f"  Eval samples: {len(eval_dataset)}")

    trainer = ContextBakingTrainer(
        model=model,
        args=baking_config,
        context_config=context_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=prompt_baking_collator,
    )

    trainer.train()

    # Evaluate baked model
    print("\n[5] Evaluating baked model...")
    baked_eval = baked_heldout = None

    if eval_qa:
        print("\n  === Baked (no system prompt) ===")
        baked_eval = evaluate_model(model, tokenizer, eval_qa, "Baked")

    if heldout_qa:
        print("\n  === Baked held-out ===")
        baked_heldout = evaluate_model(model, tokenizer, heldout_qa, "Baked (held-out)")

    # Save
    model_dir = os.path.join(baking_config.output_dir, "final_model")
    print(f"\n[6] Saving final model to {model_dir}")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    results = {"prompt_tokens": prompt_tokens}
    if baseline_none:
        results["baseline_none"] = baseline_none
    if baseline_discrete:
        results["baseline_discrete"] = baseline_discrete
    if heldout_discrete:
        results["heldout_discrete"] = heldout_discrete
    if baked_eval:
        results["baked_eval"] = baked_eval
    if baked_heldout:
        results["baked_heldout"] = baked_heldout

    results_path = os.path.join(baking_config.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Prefix context: {prompt_tokens:,} tokens -> 0 tokens at inference")

    if eval_qa and baseline_discrete and baked_eval:
        print("\nEvaluation set:")
        print(f"  Baseline (no prompt): {baseline_none['accuracy'] * 100:.1f}%")
        print(f"  Discrete prompt:      {baseline_discrete['accuracy'] * 100:.1f}%")
        print(f"  Baked:                {baked_eval['accuracy'] * 100:.1f}%")

    if heldout_qa and heldout_discrete and baked_heldout:
        print("\nHeld-out test:")
        print(f"  Discrete prompt: {heldout_discrete['accuracy'] * 100:.1f}%")
        print(f"  Baked:           {baked_heldout['accuracy'] * 100:.1f}%")

    print(f"\nOutputs saved to: {baking_config.output_dir}/")


if __name__ == "__main__":
    main()
