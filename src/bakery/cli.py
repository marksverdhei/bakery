"""Bakery CLI - train prompt-baked models from flat YAML configs.

Usage:
    bakery --config config.yaml
    bakery --config config.yaml --num_train_epochs 5 --learning_rate 1e-4
"""

import argparse
import os
import json
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
)
from peft import LoraConfig as PeftLoraConfig, get_peft_model

from bakery.config import BakeryConfig, DataConfig, LoraConfig
from bakery.trainer import PromptBakingTrainer
from bakery.data import (
    create_dataset,
    prompt_baking_collator,
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

    parser = HfArgumentParser((BakeryConfig, DataConfig, LoraConfig))

    baking_config, data_config, lora_config = parser.parse_yaml_file(
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

        override_parser = HfArgumentParser((BakeryConfig, DataConfig, LoraConfig))
        overrides = override_parser.parse_args_into_dataclasses(
            args=["--output_dir", baking_config.output_dir] + remaining_args,
            return_remaining_strings=True,
        )
        for override_cfg, base_cfg in zip(
            overrides[:3],
            (baking_config, data_config, lora_config),
        ):
            for k, v in vars(override_cfg).items():
                if k in explicit_keys:
                    setattr(base_cfg, k, v)

    # Build system prompt
    corpus = load_corpus(data_config)
    system_prompt = build_system_prompt(baking_config, data_config, corpus)
    baking_config.system_prompt = system_prompt

    # Load data
    training_prompts, precomputed_responses = load_data(data_config)
    eval_qa = load_eval_data(data_config.eval_file)
    heldout_qa = load_eval_data(data_config.heldout_file)

    print("=" * 70)
    print("Bakery - Prompt Baking with KL Divergence")
    print("=" * 70)
    print(f"  System prompt: {len(system_prompt):,} chars")
    print(f"  Training prompts: {len(training_prompts)}")
    if precomputed_responses:
        print(f"  Precomputed responses: {len(precomputed_responses)}")
    else:
        print("  Mode: on-the-fly trajectory generation")
    if eval_qa:
        print(f"  Evaluation Q&A: {len(eval_qa)}")
    if heldout_qa:
        print(f"  Held-out Q&A: {len(heldout_qa)}")

    # Load model
    print(f"\n[1] Loading model: {data_config.model_name_or_path}")
    torch_dtype = DTYPE_MAP.get(data_config.torch_dtype, torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(
        data_config.model_name_or_path,
        trust_remote_code=data_config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        # We load to CPU first with quantization, then dispatch to GPU.
        # See: https://github.com/huggingface/transformers/issues/43032
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                data_config.model_name_or_path, **load_kwargs
            )
        except torch.OutOfMemoryError:
            print("  GPU OOM during quantized loading — retrying via CPU + dispatch")
            torch.cuda.empty_cache()
            cpu_kwargs = {**load_kwargs, "device_map": "cpu"}
            base_model = AutoModelForCausalLM.from_pretrained(
                data_config.model_name_or_path, **cpu_kwargs
            )
            from accelerate import dispatch_model, infer_auto_device_map
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            device_map = infer_auto_device_map(
                base_model,
                max_memory={0: f"{gpu_mem // (1024**3) - 2}GiB", "cpu": "32GiB"},
            )
            base_model = dispatch_model(base_model, device_map=device_map)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            data_config.model_name_or_path, **load_kwargs
        )

    prompt_tokens = len(tokenizer.encode(system_prompt))
    print(f"  System prompt tokens: {prompt_tokens:,}")

    # Baseline evaluations
    if eval_qa:
        print("\n[2] Baseline evaluations...")
        print("\n  === No system prompt ===")
        baseline_none = evaluate_model(base_model, tokenizer, eval_qa, "Baseline")
        print("\n  === With system prompt (discrete) ===")
        baseline_discrete = evaluate_model(
            base_model,
            tokenizer,
            eval_qa,
            "Discrete prompt",
            system_prompt=system_prompt,
        )
        heldout_discrete = None
        if heldout_qa:
            print("\n  === Held-out (discrete prompt) ===")
            heldout_discrete = evaluate_model(
                base_model,
                tokenizer,
                heldout_qa,
                "Discrete (held-out)",
                system_prompt=system_prompt,
            )
    else:
        baseline_none = baseline_discrete = heldout_discrete = None

    # Add LoRA adapters
    print("\n[3] Adding LoRA adapters...")
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
    if precomputed_responses:
        print(
            f"  Using {len(precomputed_responses)} precomputed (prompt, response) pairs"
        )
    else:
        print(
            f"  Generating {baking_config.num_trajectories} trajectories per prompt on-the-fly"
        )

    train_dataset = create_dataset(training_prompts, precomputed_responses)

    trainer = PromptBakingTrainer(
        model=model,
        args=baking_config,
        train_dataset=train_dataset,
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
    print(f"System prompt: {prompt_tokens:,} tokens -> 0 tokens at inference")

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
