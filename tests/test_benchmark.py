"""Benchmark training step throughput for prompt baking.

Measures wall-clock time, tokens/sec, and samples/sec using a tiny GPT-2
model with LoRA. Uses precomputed responses so trajectory generation
(I/O-bound) does not dominate the measurement.

Run:
    pytest tests/test_benchmark.py -v -s
    pytest tests/test_benchmark.py -v -s -k "batch_1"
"""

import time

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig as PeftLoraConfig, get_peft_model

from bakery.config import BakeryConfig
from bakery.data import create_dataset, prompt_baking_collator
from bakery.trainer import PromptBakingTrainer


CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "{{ m['role'] }}: {{ m['content'] }}\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}assistant: {% endif %}"
)

SAMPLE_PAIRS = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Explain photosynthesis in simple terms.", "Photosynthesis is the process by which plants convert sunlight into energy."),
    ("Write a haiku about the ocean.", "Waves crash on the shore\nSalt air fills the morning breeze\nSea meets sky in blue"),
    ("What are the benefits of exercise?", "Exercise improves cardiovascular health, boosts mood, and increases energy levels."),
    ("How does a computer work?", "A computer processes instructions using a CPU, stores data in memory, and uses input/output devices."),
    ("What is machine learning?", "Machine learning is a subset of AI where systems learn patterns from data without explicit programming."),
    ("Describe the water cycle.", "Water evaporates from surfaces, forms clouds through condensation, and returns as precipitation."),
    ("What is gravity?", "Gravity is a fundamental force that attracts objects with mass toward each other."),
]


def _make_benchmark_trainer(batch_size=1):
    """Create a PromptBakingTrainer with tiny GPT-2 + LoRA for benchmarking."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = CHAT_TEMPLATE

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    peft_config = PeftLoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["c_attn"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    prompts = [p for p, _ in SAMPLE_PAIRS]
    responses = [r for _, r in SAMPLE_PAIRS]

    args = BakeryConfig(
        output_dir="/tmp/bakery_benchmark",
        system_prompt="You are a helpful assistant.",
        num_trajectories=1,
        trajectory_length=16,
        per_device_train_batch_size=batch_size,
        num_train_epochs=1,
        logging_steps=1,
        report_to="none",
        use_cpu=True,
        max_steps=1,
    )

    dataset = create_dataset(prompts, responses)

    trainer = PromptBakingTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=prompt_baking_collator,
    )
    return trainer, tokenizer


def _count_tokens(tokenizer, pairs, system_prompt="You are a helpful assistant."):
    """Count total response tokens in the benchmark data."""
    total = 0
    for _, response in pairs:
        total += len(tokenizer.encode(response))
    return total


def _run_steps(trainer, n_steps, batch_size):
    """Run n training steps via compute_loss and backward, return timing info."""
    model = trainer.model
    model.train()

    dataloader = trainer.get_train_dataloader()
    batches = []
    for batch in dataloader:
        batches.append(batch)
        if len(batches) >= n_steps + 1:
            break

    # Repeat batches if dataset is smaller than requested steps
    while len(batches) < n_steps + 1:
        batches = batches + batches
    batches = batches[: n_steps + 1]

    # Warm-up step (not timed)
    loss = trainer.compute_loss(model, batches[0])
    loss.backward()
    model.zero_grad()

    step_times = []
    forward_times = []
    backward_times = []

    for batch in batches[1:]:
        t0 = time.perf_counter()

        t_fwd_start = time.perf_counter()
        loss = trainer.compute_loss(model, batch)
        t_fwd_end = time.perf_counter()

        t_bwd_start = time.perf_counter()
        loss.backward()
        t_bwd_end = time.perf_counter()

        model.zero_grad()
        t1 = time.perf_counter()

        step_times.append(t1 - t0)
        forward_times.append(t_fwd_end - t_fwd_start)
        backward_times.append(t_bwd_end - t_bwd_start)

    return step_times, forward_times, backward_times


def _print_results(label, step_times, forward_times, backward_times, tokens_per_step, samples_per_step):
    """Print benchmark results for a configuration."""
    avg_step = sum(step_times) / len(step_times)
    avg_fwd = sum(forward_times) / len(forward_times)
    avg_bwd = sum(backward_times) / len(backward_times)

    tokens_per_sec = tokens_per_step / avg_step if avg_step > 0 else 0
    samples_per_sec = samples_per_step / avg_step if avg_step > 0 else 0

    print(f"\n  {label}:")
    print(f"    Steps:          {len(step_times)}")
    print(f"    Avg step:       {avg_step * 1000:.1f} ms")
    print(f"    Avg forward:    {avg_fwd * 1000:.1f} ms ({avg_fwd / avg_step * 100:.0f}%)")
    print(f"    Avg backward:   {avg_bwd * 1000:.1f} ms ({avg_bwd / avg_step * 100:.0f}%)")
    print(f"    Tokens/sec:     {tokens_per_sec:.1f}")
    print(f"    Samples/sec:    {samples_per_sec:.2f}")

    return {
        "avg_step_ms": avg_step * 1000,
        "avg_forward_ms": avg_fwd * 1000,
        "avg_backward_ms": avg_bwd * 1000,
        "tokens_per_sec": tokens_per_sec,
        "samples_per_sec": samples_per_sec,
    }


N_STEPS = 10


@pytest.mark.benchmark
class TestBenchmarkTraining:
    """Benchmark suite for prompt baking training throughput."""

    def test_benchmark_batch_1(self):
        """Benchmark training with batch_size=1."""
        batch_size = 1
        trainer, tokenizer = _make_benchmark_trainer(batch_size=batch_size)
        tokens_per_step = _count_tokens(tokenizer, SAMPLE_PAIRS[:batch_size])

        step_times, fwd_times, bwd_times = _run_steps(trainer, N_STEPS, batch_size)

        results = _print_results(
            f"batch_size={batch_size}", step_times, fwd_times, bwd_times,
            tokens_per_step, batch_size,
        )
        assert results["tokens_per_sec"] > 0
        assert results["samples_per_sec"] > 0

    def test_benchmark_batch_2(self):
        """Benchmark training with batch_size=2."""
        batch_size = 2
        trainer, tokenizer = _make_benchmark_trainer(batch_size=batch_size)
        tokens_per_step = _count_tokens(tokenizer, SAMPLE_PAIRS[:batch_size])

        step_times, fwd_times, bwd_times = _run_steps(trainer, N_STEPS, batch_size)

        results = _print_results(
            f"batch_size={batch_size}", step_times, fwd_times, bwd_times,
            tokens_per_step, batch_size,
        )
        assert results["tokens_per_sec"] > 0
        assert results["samples_per_sec"] > 0

    def test_benchmark_batch_4(self):
        """Benchmark training with batch_size=4."""
        batch_size = 4
        trainer, tokenizer = _make_benchmark_trainer(batch_size=batch_size)
        tokens_per_step = _count_tokens(tokenizer, SAMPLE_PAIRS[:batch_size])

        step_times, fwd_times, bwd_times = _run_steps(trainer, N_STEPS, batch_size)

        results = _print_results(
            f"batch_size={batch_size}", step_times, fwd_times, bwd_times,
            tokens_per_step, batch_size,
        )
        assert results["tokens_per_sec"] > 0
        assert results["samples_per_sec"] > 0

    def test_benchmark_summary(self, capsys):
        """Run all batch sizes and print a comparison table."""
        print("\n" + "=" * 60)
        print("BENCHMARK: Prompt Baking Training Throughput (GPT-2 + LoRA)")
        print("=" * 60)

        all_results = {}
        for batch_size in [1, 2, 4]:
            trainer, tokenizer = _make_benchmark_trainer(batch_size=batch_size)
            tokens_per_step = _count_tokens(tokenizer, SAMPLE_PAIRS[:batch_size])
            step_times, fwd_times, bwd_times = _run_steps(trainer, N_STEPS, batch_size)
            results = _print_results(
                f"batch_size={batch_size}", step_times, fwd_times, bwd_times,
                tokens_per_step, batch_size,
            )
            all_results[batch_size] = results

        print("\n" + "-" * 60)
        print(f"  {'Batch':>6}  {'Step (ms)':>10}  {'Fwd (ms)':>10}  {'Bwd (ms)':>10}  {'Tok/s':>8}  {'Samp/s':>8}")
        print(f"  {'------':>6}  {'--------':>10}  {'--------':>10}  {'--------':>10}  {'-----':>8}  {'------':>8}")
        for bs, r in all_results.items():
            print(
                f"  {bs:>6}  {r['avg_step_ms']:>10.1f}  {r['avg_forward_ms']:>10.1f}  "
                f"{r['avg_backward_ms']:>10.1f}  {r['tokens_per_sec']:>8.1f}  {r['samples_per_sec']:>8.2f}"
            )
        print("=" * 60)

        for r in all_results.values():
            assert r["tokens_per_sec"] > 0
