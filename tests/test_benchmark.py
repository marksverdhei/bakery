"""Internal benchmarking tests for prompt baking performance.

Measures key performance metrics using tiny GPT-2 + LoRA:
- Training step throughput (steps/sec)
- KL divergence computation time
- Forward pass time (teacher and student)
- Memory usage (CPU and GPU)
- Samples per second

Runs on GPU when available, CPU otherwise.
Official benchmark results should be produced on a single RTX 3090 GPU.

Usage:
    uv run pytest -m benchmark -v -s
    uv run pytest -m benchmark -v -s --benchmark-compare=main
    uv run pytest -m benchmark -v -s --benchmark-compare=abc1234
    uv run pytest -m benchmark -v -s --benchmark-save=results.json
"""

import json
import os
import shutil
import statistics
import subprocess
import tempfile
import time
import tracemalloc
from pathlib import Path

import pytest
import torch
from peft import LoraConfig as PeftLoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from bakery.config import BakeryConfig
from bakery.data import create_dataset, prompt_baking_collator
from bakery.kl import compute_kl_divergence, disable_adapters
from bakery.trainer import PromptBakingTrainer


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

USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"

# Rough upper-bound baselines (seconds) for regression detection.
# Intentionally generous to avoid flaky failures.
BASELINES = {
    "kl_divergence_single": 0.05,
    "forward_pass_teacher": 1.0,
    "forward_pass_student": 1.0,
    "compute_loss_single": 3.0,
    "compute_loss_batch": 5.0,
    "training_step_single": 8.0,
    "multi_step_3": 25.0,
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
    """Create a tiny GPT-2 model with LoRA adapter on the active device."""
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    if USE_GPU:
        model = model.to(DEVICE)
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
        use_cpu=not USE_GPU,
    )

    dataset = create_dataset(prompts, responses)
    return PromptBakingTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=prompt_baking_collator,
    )


def _sync_gpu():
    """Synchronize GPU before timing to avoid overlapping async ops."""
    if USE_GPU:
        torch.cuda.synchronize()


def _record(name, elapsed, iterations=1, extra=None):
    """Record a benchmark result."""
    entry = {
        "name": name,
        "elapsed_s": elapsed,
        "iterations": iterations,
        "per_iter_s": elapsed / iterations,
        "throughput": iterations / elapsed if elapsed > 0 else float("inf"),
        "device": DEVICE,
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
    vocab_size = 50257
    seq_len = 32
    batch = 1
    n_iters = 20

    teacher_logits = torch.randn(batch, seq_len, vocab_size, device=DEVICE)
    student_logits = torch.randn(
        batch, seq_len, vocab_size, device=DEVICE, requires_grad=True
    )
    mask = torch.ones(batch, seq_len, device=DEVICE)

    for _ in range(3):
        loss = compute_kl_divergence(teacher_logits, student_logits, mask, 1.0)
        loss.backward()
        student_logits.grad = None

    times = []
    for _ in range(n_iters):
        _sync_gpu()
        t0 = time.perf_counter()
        loss = compute_kl_divergence(teacher_logits, student_logits, mask, 1.0)
        loss.backward()
        _sync_gpu()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        student_logits.grad = None

    median = statistics.median(times)
    _record(
        "kl_divergence_single",
        median,
        1,
        {"median_ms": median * 1000, "stdev_ms": statistics.stdev(times) * 1000},
    )

    assert median < BASELINES["kl_divergence_single"]


@pytest.mark.benchmark
def test_bench_forward_pass_teacher():
    """Measure teacher forward pass time (adapters disabled)."""
    tokenizer = _make_tokenizer()
    model = _make_model()
    model.eval()

    text = "system: You are helpful.\nuser: What is 2+2?\nassistant: The answer is 4."
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(DEVICE)

    n_iters = 5
    with torch.no_grad():
        with disable_adapters(model):
            model(**inputs)

    times = []
    for _ in range(n_iters):
        _sync_gpu()
        t0 = time.perf_counter()
        with torch.no_grad():
            with disable_adapters(model):
                model(**inputs)
        _sync_gpu()
        times.append(time.perf_counter() - t0)

    median = statistics.median(times)
    _record("forward_pass_teacher", median, 1, {"median_ms": median * 1000})
    assert median < BASELINES["forward_pass_teacher"]


@pytest.mark.benchmark
def test_bench_forward_pass_student():
    """Measure student forward pass time (adapters enabled, with grad)."""
    tokenizer = _make_tokenizer()
    model = _make_model()
    model.train()

    text = "user: What is 2+2?\nassistant: The answer is 4."
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(DEVICE)

    n_iters = 5
    model(**inputs)

    times = []
    for _ in range(n_iters):
        model.zero_grad()
        _sync_gpu()
        t0 = time.perf_counter()
        out = model(**inputs)
        out.logits.sum().backward()
        _sync_gpu()
        times.append(time.perf_counter() - t0)

    median = statistics.median(times)
    _record("forward_pass_student", median, 1, {"median_ms": median * 1000})
    assert median < BASELINES["forward_pass_student"]


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

    loss = trainer.compute_loss(trainer.model, inputs)
    loss.backward()
    trainer.model.zero_grad()

    n_iters = 3
    times = []
    for _ in range(n_iters):
        trainer.model.zero_grad()
        _sync_gpu()
        t0 = time.perf_counter()
        loss = trainer.compute_loss(trainer.model, inputs)
        loss.backward()
        _sync_gpu()
        times.append(time.perf_counter() - t0)

    median = statistics.median(times)
    _record(
        "compute_loss_single",
        median,
        1,
        {"median_ms": median * 1000, "loss_value": loss.item()},
    )
    assert median < BASELINES["compute_loss_single"]


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

    loss = trainer.compute_loss(trainer.model, inputs)
    loss.backward()
    trainer.model.zero_grad()

    n_iters = 3
    times = []
    for _ in range(n_iters):
        trainer.model.zero_grad()
        _sync_gpu()
        t0 = time.perf_counter()
        loss = trainer.compute_loss(trainer.model, inputs)
        loss.backward()
        _sync_gpu()
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
    assert median < BASELINES["compute_loss_batch"]


@pytest.mark.benchmark
def test_bench_training_step_single():
    """Measure a single full forward+backward pass with trajectory generation."""
    trainer = _make_trainer(
        prompts=["What is 2+2?"],
        responses=["placeholder"],
    )

    trainer.model.eval()
    with torch.no_grad():
        response = trainer._generate_trajectory("What is 2+2?")
    trainer.model.train()
    inputs = {"user_messages": ["What is 2+2?"], "responses": [response]}
    loss = trainer.compute_loss(trainer.model, inputs)
    loss.backward()
    trainer.model.zero_grad()

    _sync_gpu()
    t0 = time.perf_counter()
    trainer.model.eval()
    with torch.no_grad():
        response = trainer._generate_trajectory("What is 2+2?")
    trainer.model.train()
    inputs = {"user_messages": ["What is 2+2?"], "responses": [response]}
    loss = trainer.compute_loss(trainer.model, inputs)
    loss.backward()
    _sync_gpu()
    elapsed = time.perf_counter() - t0

    _record("training_step_single", elapsed, 1, {"elapsed_ms": elapsed * 1000})
    assert elapsed < BASELINES["training_step_single"]


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

    loss = trainer.compute_loss(trainer.model, dict(inputs))
    loss.backward()
    trainer.model.zero_grad()

    _sync_gpu()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        trainer.model.zero_grad()
        loss = trainer.compute_loss(trainer.model, dict(inputs))
        loss.backward()
    _sync_gpu()
    elapsed = time.perf_counter() - t0

    steps_per_sec = n_steps / elapsed
    samples_per_sec = (n_steps * 2) / elapsed
    _record(
        "multi_step_3",
        elapsed,
        n_steps,
        {"steps_per_sec": steps_per_sec, "samples_per_sec": samples_per_sec},
    )
    assert elapsed < BASELINES["multi_step_3"]


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

    if USE_GPU:
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

    if USE_GPU:
        gpu_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        extra["gpu_peak_mb"] = gpu_peak

    _record("memory_compute_loss_batch4", 0, 1, extra)
    assert peak_mem / (1024 * 1024) < 2048


# ---------------------------------------------------------------------------
# GPU benchmarks (Qwen 3.5 9B QLoRA, requires CUDA)
# ---------------------------------------------------------------------------

# Target: ~18-20 GB VRAM on RTX 3090.
# flash-linear-attention + causal-conv1d are auto-installed if missing.
# batch=2 with long responses accounts for double forward pass (teacher + student).
GPU_MODEL = "Qwen/Qwen3.5-9B"
GPU_BATCH_SIZE = 2

_GPU_PROMPTS = [
    "Write a detailed essay about the history of artificial intelligence, "
    "covering its origins in the 1950s, the AI winters, the rise of machine "
    "learning, deep learning breakthroughs, and modern large language models. "
    "Include key figures, milestones, and the societal implications of each era.",
] * GPU_BATCH_SIZE

_GPU_RESPONSES = [
    "Artificial intelligence has a rich and complex history spanning over seven "
    "decades. The field began in the summer of 1956 at Dartmouth College, where "
    "John McCarthy, Marvin Minsky, Nathaniel Rochester, and Claude Shannon "
    "organized a workshop that would define the field. Early AI research was "
    "characterized by unbridled optimism and significant government funding. "
    "Researchers believed that human-level AI was just around the corner. "
    "Programs like the Logic Theorist and the General Problem Solver showed "
    "that machines could perform tasks previously thought to require human "
    "intelligence. However, the limitations of these early systems soon became "
    "apparent, leading to the first AI winter in the 1970s. Funding dried up "
    "and progress stalled for nearly a decade. The second wave came with expert "
    "systems in the 1980s, which showed commercial promise but ultimately proved "
    "brittle and expensive to maintain. Machine learning emerged as a more "
    "robust approach, with neural networks gaining traction after Rumelhart "
    "and Hinton popularized backpropagation. The deep learning revolution began "
    "in 2012 when AlexNet won ImageNet, and since then transformers, attention "
    "mechanisms, and large language models have transformed the field entirely.",
] * GPU_BATCH_SIZE


def _make_gpu_model():
    """Load Qwen 3.5 9B with QLoRA for GPU benchmarking."""
    from transformers import AutoConfig, BitsAndBytesConfig
    from bakery.deps import ensure_deps

    model_config = AutoConfig.from_pretrained(GPU_MODEL)
    ensure_deps(model_type=model_config.model_type, features=["qlora"])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        GPU_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    peft_config = PeftLoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.enable_input_require_grads()
    return model


def _make_gpu_trainer():
    """Create a PromptBakingTrainer with Qwen 3.5 9B QLoRA."""
    tokenizer = AutoTokenizer.from_pretrained(GPU_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _make_gpu_model()

    args = BakeryConfig(
        output_dir="/tmp/bakery_bench_gpu",
        system_prompt="You are a helpful assistant.",
        num_trajectories=1,
        trajectory_length=16,
        per_device_train_batch_size=GPU_BATCH_SIZE,
        num_train_epochs=1,
        logging_steps=1,
        report_to="none",
        gradient_checkpointing=True,
        bf16=True,
    )

    dataset = create_dataset(_GPU_PROMPTS, _GPU_RESPONSES)
    return PromptBakingTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=prompt_baking_collator,
    )


# Cache the GPU trainer so both tests share the same model in VRAM.
_gpu_trainer_cache = None


def _get_gpu_trainer():
    global _gpu_trainer_cache
    if _gpu_trainer_cache is None:
        _gpu_trainer_cache = _make_gpu_trainer()
    return _gpu_trainer_cache


@pytest.mark.benchmark
@pytest.mark.gpu
def test_bench_gpu_compute_loss():
    """Measure compute_loss on Qwen 3.5 9B QLoRA (batch=2)."""
    if not USE_GPU:
        pytest.skip("GPU not available")

    trainer = _get_gpu_trainer()
    inputs = {
        "user_messages": _GPU_PROMPTS,
        "responses": _GPU_RESPONSES,
    }

    # Warm up
    loss = trainer.compute_loss(trainer.model, inputs)
    loss.backward()
    trainer.model.zero_grad()
    torch.cuda.synchronize()

    n_iters = 3
    times = []
    for _ in range(n_iters):
        trainer.model.zero_grad()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = trainer.compute_loss(trainer.model, inputs)
        loss.backward()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    median = statistics.median(times)
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    _record(
        "gpu_compute_loss_9b",
        median,
        1,
        {
            "median_ms": median * 1000,
            "batch_size": GPU_BATCH_SIZE,
            "samples_per_sec": GPU_BATCH_SIZE / median,
            "gpu_peak_gb": peak_gb,
        },
    )


@pytest.mark.benchmark
@pytest.mark.gpu
def test_bench_gpu_multi_step():
    """Measure 3 training steps on Qwen 3.5 9B QLoRA."""
    if not USE_GPU:
        pytest.skip("GPU not available")

    n_steps = 3
    trainer = _get_gpu_trainer()
    inputs = {
        "user_messages": _GPU_PROMPTS,
        "responses": _GPU_RESPONSES,
    }

    # Warm up
    loss = trainer.compute_loss(trainer.model, dict(inputs))
    loss.backward()
    trainer.model.zero_grad()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        trainer.model.zero_grad()
        loss = trainer.compute_loss(trainer.model, dict(inputs))
        loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    _record(
        "gpu_multi_step_9b",
        elapsed,
        n_steps,
        {
            "steps_per_sec": n_steps / elapsed,
            "samples_per_sec": (n_steps * GPU_BATCH_SIZE) / elapsed,
            "gpu_peak_gb": peak_gb,
        },
    )


# ---------------------------------------------------------------------------
# Comparison & Summary
# ---------------------------------------------------------------------------


def _run_baseline_in_worktree(ref: str) -> dict:
    """Run benchmarks at a given git ref using a temporary worktree.

    Copies the *current* benchmark test and conftest into the worktree so both
    baseline and current runs use the same test code (including GPU support).
    Only the bakery source code differs between the two runs.

    Parses the BENCHMARK SUMMARY table from stdout to extract results.
    Returns a dict mapping benchmark name to result dict.
    """
    repo_root = Path(__file__).resolve().parent.parent
    worktree_dir = tempfile.mkdtemp(prefix=f"bakery_bench_{ref[:8]}_")

    try:
        subprocess.run(
            ["git", "worktree", "add", "--detach", worktree_dir, ref],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )

        # Copy current benchmark test + conftest into worktree so both runs
        # use identical test code (GPU support, timing, etc.)
        tests_dir = os.path.join(worktree_dir, "tests")
        os.makedirs(tests_dir, exist_ok=True)
        shutil.copy2(
            os.path.join(repo_root, "tests", "test_benchmark.py"),
            os.path.join(tests_dir, "test_benchmark.py"),
        )
        conftest_src = os.path.join(repo_root, "tests", "conftest.py")
        if os.path.exists(conftest_src):
            shutil.copy2(conftest_src, os.path.join(tests_dir, "conftest.py"))

        # Sync deps in worktree (source may have changed)
        subprocess.run(
            ["uv", "sync"],
            cwd=worktree_dir,
            capture_output=True,
            timeout=120,
        )

        result = subprocess.run(
            [
                "uv",
                "run",
                "pytest",
                "tests/test_benchmark.py",
                "-m",
                "benchmark",
                "-v",
                "-s",
                "-p",
                "no:cacheprovider",
            ],
            cwd=worktree_dir,
            capture_output=True,
            text=True,
            timeout=300,
        )

        return _parse_benchmark_stdout(result.stdout)
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", worktree_dir],
            cwd=repo_root,
            capture_output=True,
        )
        if os.path.exists(worktree_dir):
            shutil.rmtree(worktree_dir, ignore_errors=True)

    return {}


def _parse_benchmark_stdout(stdout: str) -> dict:
    """Parse benchmark results from pytest stdout output.

    Looks for lines in the BENCHMARK SUMMARY table with format:
        test_name          123.4  45.67 iter/s       OK
    """
    results = {}
    in_table = False
    past_header = False
    for line in stdout.splitlines():
        if "BENCHMARK SUMMARY" in line:
            in_table = True
            continue
        if in_table and past_header and line.startswith("="):
            break
        if in_table and line.startswith("-"):
            past_header = True
            continue
        if not in_table or not past_header:
            continue
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            name = parts[0]
            try:
                time_ms = float(parts[1])
                results[name] = {"elapsed_s": time_ms / 1000}
            except ValueError:
                continue
    return results


def _format_delta(current_ms, baseline_ms):
    """Format a time delta as a string.

    Negative time = faster (good), shown as positive speedup percentage.
    """
    if baseline_ms == 0:
        return ""
    ratio = current_ms / baseline_ms
    if ratio < 0.95:
        speedup = (1 - ratio) * 100
        return f"{speedup:.1f}% faster"
    elif ratio > 1.05:
        slowdown = (ratio - 1) * 100
        return f"{slowdown:.1f}% SLOWER"
    else:
        return "~same"


def _format_throughput(r):
    """Format throughput from a result dict."""
    if "steps_per_sec" in r:
        return f"{r['steps_per_sec']:.2f} step/s"
    elif "samples_per_sec" in r:
        return f"{r['samples_per_sec']:.2f} samp/s"
    elif r.get("throughput") and r["elapsed_s"] > 0:
        return f"{r['throughput']:.2f} iter/s"
    return ""


@pytest.mark.benchmark
def test_bench_print_summary(request):
    """Print a summary table of all benchmark results (must run last).

    With --benchmark-compare=REF, runs benchmarks at REF in a worktree
    and shows a comparison table.
    """
    if not _benchmark_results:
        pytest.skip("No benchmark results collected")

    # Save results if requested
    save_path = request.config.getoption("--benchmark-save", default=None)
    if save_path:
        save_data = {}
        for r in _benchmark_results:
            save_data[r["name"]] = {
                k: v
                for k, v in r.items()
                if isinstance(v, (int, float, str)) and k != "name"
            }
        save_data["_meta"] = {
            "device": DEVICE,
            "gpu": torch.cuda.get_device_name(0) if USE_GPU else None,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(save_path, "w") as f:
            json.dump(save_data, f, indent=2)

    # Load baseline for comparison
    compare_ref = request.config.getoption("--benchmark-compare", default=None)
    baseline_data = {}
    if compare_ref:
        print(f"\nRunning baseline benchmarks at '{compare_ref}' via worktree...")
        baseline_data = _run_baseline_in_worktree(compare_ref)

    # Build summary
    lines = []
    lines.append("")
    lines.append("=" * 95)
    lines.append(
        f"BENCHMARK SUMMARY  (device: {DEVICE}"
        + (f", {torch.cuda.get_device_name(0)}" if USE_GPU else "")
        + ")"
    )
    lines.append("=" * 95)

    if baseline_data:
        lines.append(
            f"{'Test':<30} {'Current':>10} {'Baseline':>10} "
            f"{'Delta':>16} {'Throughput':>14}"
        )
    else:
        lines.append(f"{'Test':<30} {'Time (ms)':>10} {'Throughput':>14} {'Status':>8}")
    lines.append("-" * 95)

    for r in _benchmark_results:
        name = r["name"]
        if r["elapsed_s"] == 0:
            cur_ms_str = "N/A"
            cur_ms = 0
        else:
            cur_ms = r["elapsed_s"] * 1000
            cur_ms_str = f"{cur_ms:>9.1f}"

        tp = _format_throughput(r)

        if baseline_data and name in baseline_data:
            base_ms = baseline_data[name].get("elapsed_s", 0) * 1000
            base_str = f"{base_ms:>9.1f}" if base_ms > 0 else "N/A"
            delta = _format_delta(cur_ms, base_ms) if cur_ms > 0 and base_ms > 0 else ""
            lines.append(
                f"{name:<30} {cur_ms_str:>10} {base_str:>10} {delta:>16} {tp:>14}"
            )
        else:
            baseline = BASELINES.get(name)
            status = ""
            if baseline and r["elapsed_s"] > 0:
                status = "OK" if r["elapsed_s"] < baseline else "SLOW"
            lines.append(f"{name:<30} {cur_ms_str:>10} {tp:>14} {status:>8}")

    # Memory info
    for r in _benchmark_results:
        if "cpu_peak_mb" in r:
            lines.append("")
            lines.append(f"  CPU peak memory: {r['cpu_peak_mb']:.1f} MB")
            if "gpu_peak_mb" in r:
                lines.append(f"  GPU peak memory: {r['gpu_peak_mb']:.1f} MB")

    if baseline_data:
        meta = baseline_data.get("_meta", {})
        lines.append("")
        lines.append(f"  Baseline: {compare_ref} (device: {meta.get('device', '?')})")

    lines.append("=" * 95)
    summary = "\n".join(lines)
    print(summary)
