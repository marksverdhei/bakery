"""Bakery CLI - train prompt-baked models from flat YAML configs.

Usage:
    bakery --config config.yaml
    bakery --config config.yaml --num_train_epochs 5 --learning_rate 1e-4
"""

import os
import sys
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
    parser = HfArgumentParser((BakeryConfig, DataConfig, LoraConfig))

    config_file = None
    remaining_args = []
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == "--config" and i + 1 < len(argv):
            config_file = argv[i + 1]
            i += 2
        elif argv[i].startswith("--config="):
            config_file = argv[i].split("=", 1)[1]
            i += 1
        else:
            remaining_args.append(argv[i])
            i += 1

    if config_file:
        baking_config, data_config, lora_config = parser.parse_yaml_file(
            config_file, allow_extra_keys=True
        )
        # Apply CLI overrides on top of YAML config
        if remaining_args:
            override_parser = HfArgumentParser((BakeryConfig, DataConfig, LoraConfig))
            overrides = override_parser.parse_args_into_dataclasses(
                args=remaining_args, return_remaining_strings=True
            )
            for override_cfg, base_cfg in zip(
                overrides[:3], (baking_config, data_config, lora_config)
            ):
                for k, v in vars(override_cfg).items():
                    default = type(base_cfg).__dataclass_fields__.get(k)
                    if default is not None and v != default.default:
                        setattr(base_cfg, k, v)
    else:
        baking_config, data_config, lora_config = parser.parse_args_into_dataclasses()

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
    )
    if data_config.load_in_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=data_config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=data_config.bnb_4bit_use_double_quant,
        )
        print("  Loading in 4-bit (QLoRA mode)")

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
