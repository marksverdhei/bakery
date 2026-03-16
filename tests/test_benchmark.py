"""Internal benchmarking tests for prompt baking performance.

Measures key performance metrics using tiny GPT-2 + LoRA on CPU:
- Training step throughput (steps/sec)
- KL divergence computation time
- Forward pass time (teacher and student)
- Memory usage (GPU only, when available)
- Samples per second

Run selectively with: uv run pytest -m benchmark -v
"""

import time
import statistics
import tracemalloc

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig as PeftLoraConfig, get_peft_model

from bakery.config import BakeryConfig
from bakery.data import create_dataset, prompt_baking_collator
from bakery.trainer import PromptBakingTrainer
from bakery.kl import compute_kl_divergence, disable_adapters


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "{{ m['role'] }}: {{ m['content'] }}\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}assistant: {% endif %}"
)

SAMPLE_PROMPTS = [
    "What is 2+2?",
    "Explain gravity in one sentence.",
    "Who wrote Hamlet?",
    "Name three primary colors.",
]

SAMPLE_RESPONSES = [
    "The answer is 4.",
    "Gravity is the force that attracts objects with mass toward each other.",
    "William Shakespeare wrote Hamlet.",
    "Red, blue, and yellow are three primary colors.",
]

# Rough upper-bound baselines (seconds) for regression detection on CPU.
# These are intentionally generous to avoid flaky failures; the goal is to
# catch large regressions (e.g. 5x slowdowns), not micro-optimizations.
BASELINES = {
    "kl_divergence_single": 0.05,  # single KL computation
    "forward_pass_teacher": 1.0,  # one forward pass, adapters disabled
    "forward_pass_student": 1.0,  # one forward pass, adapters enabled
    "compute_loss_single": 3.0,  # single compute_loss call
    "compute_loss_batch": 5.0,  # batched compute_loss call
    "training_step_single": 8.0,  # single training_step (with generation)
    "multi_step_3": 25.0,  # 3 training steps end-to-end
}

# Accumulates results across the test session for the summary.
_benchmark_results: list[dict] = []


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _make_tokenizer():
    """Create a GPT-2 tokenizer configured for chat."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


def _make_model():
    """Create a tiny GPT-2 model with LoRA adapter."""
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    peft_config = PeftLoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["c_attn"],
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, peft_config)


def _make_trainer(prompts=None, responses=None, batch_size=1):
    """Create a PromptBakingTrainer with tiny GPT-2 + LoRA for benchmarking."""
    prompts = prompts or SAMPLE_PROMPTS[:1]
    responses = responses or SAMPLE_RESPONSES[:1]

    tokenizer = _make_tokenizer()
    model = _make_model()

    args = BakeryConfig(
        output_dir="/tmp/bakery_bench",
        system_prompt="You are a helpful assistant.",
        num_trajectories=1,
        trajectory_length=16,
        per_device_train_batch_size=batch_size,
        num_train_epochs=1,
        logging_steps=1,
        report_to="none",
        use_cpu=True,
    )

    dataset = create_dataset(prompts, responses)
    return PromptBakingTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=prompt_baking_collator,
    )


def _record(name, elapsed, iterations=1, extra=None):
    """Record a benchmark result."""
    entry = {
        "name": name,
        "elapsed_s": elapsed,
        "iterations": iterations,
        "per_iter_s": elapsed / iterations,
        "throughput": iterations / elapsed if elapsed > 0 else float("inf"),
    }
    if extra:
        entry.update(extra)
    _benchmark_results.append(entry)
    return entry


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_kl_divergence():
    """Measure raw KL divergence computation time."""
    vocab_size = 50257  # GPT-2 vocab
    seq_len = 32
    batch = 1
    n_iters = 20

    teacher_logits = torch.randn(batch, seq_len, vocab_size)
    student_logits = torch.randn(batch, seq_len, vocab_size, requires_grad=True)
    mask = torch.ones(batch, seq_len)

    # Warm up
    for _ in range(3):
        loss = compute_kl_divergence(teacher_logits, student_logits, mask, 1.0)
        loss.backward()
        student_logits.grad = None

    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        loss = compute_kl_divergence(teacher_logits, student_logits, mask, 1.0)
        loss.backward()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        student_logits.grad = None

    median = statistics.median(times)
    result = _record("kl_divergence_single", median, 1)
    result["median_ms"] = median * 1000
    result["stdev_ms"] = statistics.stdev(times) * 1000

    assert median < BASELINES["kl_divergence_single"], (
        f"KL divergence took {median:.4f}s, "
        f"exceeds baseline {BASELINES['kl_divergence_single']}s"
    )


@pytest.mark.benchmark
def test_bench_forward_pass_teacher():
    """Measure teacher forward pass time (adapters disabled)."""
    tokenizer = _make_tokenizer()
    model = _make_model()
    model.eval()

    text = "system: You are helpful.\nuser: What is 2+2?\nassistant: The answer is 4."
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)

    n_iters = 5
    # Warm up
    with torch.no_grad():
        with disable_adapters(model):
            model(**inputs)

    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        with torch.no_grad():
            with disable_adapters(model):
                model(**inputs)
        times.append(time.perf_counter() - t0)

    median = statistics.median(times)
    _record("forward_pass_teacher", median, 1, {"median_ms": median * 1000})

    assert median < BASELINES["forward_pass_teacher"], (
        f"Teacher forward pass took {median:.4f}s, "
        f"exceeds baseline {BASELINES['forward_pass_teacher']}s"
    )


@pytest.mark.benchmark
def test_bench_forward_pass_student():
    """Measure student forward pass time (adapters enabled, with grad)."""
    tokenizer = _make_tokenizer()
    model = _make_model()
    model.train()

    text = "user: What is 2+2?\nassistant: The answer is 4."
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)

    n_iters = 5
    # Warm up
    model(**inputs)

    times = []
    for _ in range(n_iters):
        model.zero_grad()
        t0 = time.perf_counter()
        out = model(**inputs)
        out.logits.sum().backward()
        times.append(time.perf_counter() - t0)

    median = statistics.median(times)
    _record("forward_pass_student", median, 1, {"median_ms": median * 1000})

    assert median < BASELINES["forward_pass_student"], (
        f"Student forward pass took {median:.4f}s, "
        f"exceeds baseline {BASELINES['forward_pass_student']}s"
    )


@pytest.mark.benchmark
def test_bench_compute_loss_single():
    """Measure compute_loss with a single prompt-response pair."""
    trainer = _make_trainer(
        prompts=["What is 2+2?"],
        responses=["The answer is 4."],
    )
    inputs = {
        "user_messages": ["What is 2+2?"],
        "responses": ["The answer is 4."],
    }

    # Warm up
    loss = trainer.compute_loss(trainer.model, inputs)
    loss.backward()
    trainer.model.zero_grad()

    n_iters = 3
    times = []
    for _ in range(n_iters):
        trainer.model.zero_grad()
        t0 = time.perf_counter()
        loss = trainer.compute_loss(trainer.model, inputs)
        loss.backward()
        times.append(time.perf_counter() - t0)

    median = statistics.median(times)
    _record(
        "compute_loss_single",
        median,
        1,
        {"median_ms": median * 1000, "loss_value": loss.item()},
    )

    assert median < BASELINES["compute_loss_single"], (
        f"compute_loss (single) took {median:.4f}s, "
        f"exceeds baseline {BASELINES['compute_loss_single']}s"
    )


@pytest.mark.benchmark
def test_bench_compute_loss_batch():
    """Measure compute_loss with a batch of 4 prompt-response pairs."""
    trainer = _make_trainer(
        prompts=SAMPLE_PROMPTS,
        responses=SAMPLE_RESPONSES,
        batch_size=4,
    )
    inputs = {
        "user_messages": SAMPLE_PROMPTS,
        "responses": SAMPLE_RESPONSES,
    }

    # Warm up
    loss = trainer.compute_loss(trainer.model, inputs)
    loss.backward()
    trainer.model.zero_grad()

    n_iters = 3
    times = []
    for _ in range(n_iters):
        trainer.model.zero_grad()
        t0 = time.perf_counter()
        loss = trainer.compute_loss(trainer.model, inputs)
        loss.backward()
        times.append(time.perf_counter() - t0)

    median = statistics.median(times)
    samples_per_sec = len(SAMPLE_PROMPTS) / median if median > 0 else float("inf")
    _record(
        "compute_loss_batch",
        median,
        1,
        {
            "median_ms": median * 1000,
            "batch_size": len(SAMPLE_PROMPTS),
            "samples_per_sec": samples_per_sec,
        },
    )

    assert median < BASELINES["compute_loss_batch"], (
        f"compute_loss (batch=4) took {median:.4f}s, "
        f"exceeds baseline {BASELINES['compute_loss_batch']}s"
    )


@pytest.mark.benchmark
def test_bench_training_step_single():
    """Measure a single full forward+backward pass with trajectory generation."""
    trainer = _make_trainer(
        prompts=["What is 2+2?"],
        responses=["placeholder"],
    )

    # Warm up: generate a trajectory then compute loss
    trainer.model.eval()
    with torch.no_grad():
        response = trainer._generate_trajectory("What is 2+2?")
    trainer.model.train()
    inputs = {"user_messages": ["What is 2+2?"], "responses": [response]}
    loss = trainer.compute_loss(trainer.model, inputs)
    loss.backward()
    trainer.model.zero_grad()

    # Timed run: trajectory generation + compute_loss + backward
    t0 = time.perf_counter()
    trainer.model.eval()
    with torch.no_grad():
        response = trainer._generate_trajectory("What is 2+2?")
    trainer.model.train()
    inputs = {"user_messages": ["What is 2+2?"], "responses": [response]}
    loss = trainer.compute_loss(trainer.model, inputs)
    loss.backward()
    elapsed = time.perf_counter() - t0

    _record(
        "training_step_single",
        elapsed,
        1,
        {
            "elapsed_ms": elapsed * 1000,
        },
    )

    assert elapsed < BASELINES["training_step_single"], (
        f"training_step took {elapsed:.4f}s, "
        f"exceeds baseline {BASELINES['training_step_single']}s"
    )


@pytest.mark.benchmark
def test_bench_multi_step():
    """Measure 3 forward+backward steps end-to-end for throughput calculation."""
    n_steps = 3
    trainer = _make_trainer(
        prompts=SAMPLE_PROMPTS[:2],
        responses=SAMPLE_RESPONSES[:2],
        batch_size=2,
    )
    inputs = {
        "user_messages": SAMPLE_PROMPTS[:2],
        "responses": SAMPLE_RESPONSES[:2],
    }

    # Warm up
    loss = trainer.compute_loss(trainer.model, dict(inputs))
    loss.backward()
    trainer.model.zero_grad()

    t0 = time.perf_counter()
    for _ in range(n_steps):
        trainer.model.zero_grad()
        loss = trainer.compute_loss(trainer.model, dict(inputs))
        loss.backward()
    elapsed = time.perf_counter() - t0

    steps_per_sec = n_steps / elapsed
    samples_per_sec = (n_steps * 2) / elapsed
    _record(
        "multi_step_3",
        elapsed,
        n_steps,
        {
            "steps_per_sec": steps_per_sec,
            "samples_per_sec": samples_per_sec,
        },
    )

    assert elapsed < BASELINES["multi_step_3"], (
        f"{n_steps} training steps took {elapsed:.4f}s, "
        f"exceeds baseline {BASELINES['multi_step_3']}s"
    )


@pytest.mark.benchmark
def test_bench_memory_tracking():
    """Measure peak CPU memory for compute_loss (GPU memory if available)."""
    trainer = _make_trainer(
        prompts=SAMPLE_PROMPTS,
        responses=SAMPLE_RESPONSES,
        batch_size=4,
    )
    inputs = {
        "user_messages": SAMPLE_PROMPTS,
        "responses": SAMPLE_RESPONSES,
    }

    gpu_available = torch.cuda.is_available()
    if gpu_available:
        torch.cuda.reset_peak_memory_stats()

    tracemalloc.start()

    loss = trainer.compute_loss(trainer.model, inputs)
    loss.backward()

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    extra = {
        "cpu_peak_mb": peak_mem / (1024 * 1024),
        "cpu_current_mb": current_mem / (1024 * 1024),
    }

    if gpu_available:
        gpu_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        extra["gpu_peak_mb"] = gpu_peak

    _record("memory_compute_loss_batch4", 0, 1, extra)

    # Sanity check: peak CPU memory should be under 2 GB for tiny GPT-2
    assert peak_mem / (1024 * 1024) < 2048, (
        f"Peak CPU memory {peak_mem / (1024 * 1024):.1f} MB exceeds 2 GB limit"
    )


# ---------------------------------------------------------------------------
# Summary (printed at session end)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_print_summary():
    """Print a summary table of all benchmark results (must run last)."""
    if not _benchmark_results:
        pytest.skip("No benchmark results collected")

    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("BENCHMARK SUMMARY")
    lines.append("=" * 80)
    lines.append(f"{'Test':<35} {'Time (ms)':>10} {'Throughput':>14} {'Status':>8}")
    lines.append("-" * 80)

    for r in _benchmark_results:
        name = r["name"]
        if r["elapsed_s"] == 0:
            time_str = "N/A"
        else:
            time_str = f"{r['elapsed_s'] * 1000:>9.1f}"

        if "steps_per_sec" in r:
            tp = f"{r['steps_per_sec']:.2f} step/s"
        elif "samples_per_sec" in r:
            tp = f"{r['samples_per_sec']:.2f} samp/s"
        elif r.get("throughput") and r["elapsed_s"] > 0:
            tp = f"{r['throughput']:.2f} iter/s"
        else:
            tp = ""

        baseline = BASELINES.get(name)
        if baseline and r["elapsed_s"] > 0:
            status = "OK" if r["elapsed_s"] < baseline else "SLOW"
        else:
            status = "-"

        lines.append(f"{name:<35} {time_str:>10} {tp:>14} {status:>8}")

    # Memory info
    for r in _benchmark_results:
        if "cpu_peak_mb" in r:
            lines.append("")
            lines.append(f"  CPU peak memory: {r['cpu_peak_mb']:.1f} MB")
            if "gpu_peak_mb" in r:
                lines.append(f"  GPU peak memory: {r['gpu_peak_mb']:.1f} MB")

    lines.append("=" * 80)
    summary = "\n".join(lines)
    print(summary)
