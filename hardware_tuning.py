#!/usr/bin/env python3
"""Hardware detection and model-server recommendations for the setup wizard."""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable


KV_BYTES = {"f16": 2.0, "q8_0": 1.0, "q4_0": 0.5}
MIB = 1024 * 1024
MODEL_LOAD_RESERVE_MIB = 256
Input = Callable[[str], str]
Report = Callable[[str], None]


def _ask_choice(
    message: str,
    options: tuple[str, ...],
    default: str,
    *,
    input_fn: Input,
    report: Report,
) -> str:
    labels = "/".join(options)
    while True:
        answer = input_fn(f"{message} ({labels}) [{default}]: ").strip().casefold()
        if not answer:
            return default
        if answer in options:
            return answer
        report(f"Choose one of: {labels}.")


def _ask_int(
    message: str,
    default: int,
    minimum: int,
    maximum: int,
    *,
    input_fn: Input,
    report: Report,
) -> int:
    while True:
        answer = input_fn(f"{message} [{default}]: ").strip()
        if not answer:
            return default
        try:
            value = int(answer)
        except ValueError:
            report("Enter a whole number.")
            continue
        if not minimum <= value <= maximum:
            report(f"Enter a value between {minimum} and {maximum}.")
            continue
        return value


def detect_memory() -> tuple[int, int]:
    """Return (available, total) system memory in MiB."""
    try:
        values: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            key, raw = line.split(":", 1)
            if key in {"MemAvailable", "MemTotal"}:
                values[key] = int(raw.split()[0]) // 1024
        if "MemAvailable" in values and "MemTotal" in values:
            return values["MemAvailable"], values["MemTotal"]
    except (OSError, ValueError, IndexError):
        pass

    page_size = os.sysconf("SC_PAGE_SIZE")
    total_pages = os.sysconf("SC_PHYS_PAGES")
    available_pages = os.sysconf("SC_AVPHYS_PAGES")
    return available_pages * page_size // MIB, total_pages * page_size // MIB


def detect_gpu() -> dict[str, Any] | None:
    """Read the first NVIDIA GPU's name and current memory usage, if available."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = result.stdout.splitlines()
    if not lines:
        return None
    first_gpu = lines[0].split(",", 2)
    if len(first_gpu) != 3:
        return None
    try:
        return {
            "name": first_gpu[0].strip(),
            "total_mib": int(first_gpu[1].strip()),
            "free_mib": int(first_gpu[2].strip()),
        }
    except ValueError:
        return None


def scan_model_metadata(model_path: Path) -> dict[str, int | str]:
    """Read GGUF metadata using llama.cpp's vocabulary-only mode, not weights."""
    from llama_cpp import Llama

    model = Llama(
        model_path=str(model_path),
        vocab_only=True,
        n_gpu_layers=0,
        n_ctx=32,
        n_batch=32,
        n_ubatch=32,
        verbose=False,
    )
    try:
        raw = model.metadata
    finally:
        model.close()

    architecture = raw.get("general.architecture")
    if not architecture:
        raise ValueError("GGUF metadata is missing general.architecture")
    prefix = f"{architecture}."

    def required_integer(key: str) -> int:
        try:
            return int(raw[f"{prefix}{key}"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"GGUF metadata is missing {prefix}{key}") from exc

    return {
        "architecture": architecture,
        "layers": required_integer("block_count"),
        "kv_heads": required_integer("attention.head_count_kv"),
        "key_length": required_integer("attention.key_length"),
        "value_length": required_integer("attention.value_length"),
        "context_length": required_integer("context_length"),
    }


def choose_recommendation(
    settings: dict[str, Any],
    *,
    model_size_mib: float,
    metadata: dict[str, int | str],
    gpu: dict[str, Any] | None,
    ram_budget_mib: int,
    resource_mode: str,
    cpu_threads: int,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Choose the largest context that fits approximate GPU/RAM budgets."""
    server = settings["server"]
    layers = int(metadata["layers"])
    max_context = int(metadata["context_length"])
    head_bytes_per_token = (
        int(metadata["kv_heads"])
        * (int(metadata["key_length"]) + int(metadata["value_length"]))
        * KV_BYTES[server["kv_type"]]
    )
    current_context = int(server["n_ctx"])
    contexts = sorted(
        {
            value
            for value in (
                1024, 2048, 4096, 8192, 16384, 32768, 65536,
                131072, max_context, current_context,
            )
            if 1024 <= value <= max_context
        },
        reverse=True,
    )

    gpu_budget_mib = 0.0
    if gpu is not None:
        reserve = max(256, gpu["total_mib"] * 0.10)
        gpu_budget_mib = max(0.0, gpu["free_mib"] - reserve)

    for context in contexts:
        kv_mib = layers * head_bytes_per_token * context / MIB
        if resource_mode == "gpu-only":
            gpu_layers = layers
            cpu_weights_mib = 0.0
            gpu_weights_mib = model_size_mib * 1.10
        elif resource_mode == "cpu-only" or gpu is None:
            gpu_layers = 0
            cpu_weights_mib = model_size_mib
            gpu_weights_mib = 0.0
        else:
            estimated_gpu_total_mib = model_size_mib * 1.10 + kv_mib
            gpu_layers = min(
                layers,
                int(layers * gpu_budget_mib / max(estimated_gpu_total_mib, 1)),
            )
            gpu_weights_mib = model_size_mib * 1.10 * gpu_layers / layers
            cpu_weights_mib = model_size_mib * (layers - gpu_layers) / layers

        gpu_kv_mib = kv_mib * gpu_layers / layers
        cpu_kv_mib = kv_mib - gpu_kv_mib
        cpu_required_mib = cpu_weights_mib + cpu_kv_mib + MODEL_LOAD_RESERVE_MIB
        if gpu_layers and gpu_weights_mib + gpu_kv_mib > gpu_budget_mib:
            continue
        if cpu_required_mib > ram_budget_mib:
            continue
        if resource_mode == "gpu-only" and gpu is None:
            continue
        if resource_mode == "gpu+cpu" and gpu is not None and gpu_layers == 0:
            continue

        result = {
            "n_gpu_layers": -1 if gpu_layers == layers and gpu is not None else gpu_layers,
            "n_ctx": context,
            "n_threads": cpu_threads,
            "n_threads_batch": cpu_threads,
            "kv_mib_estimate": round(kv_mib),
            "gpu_mib_estimate": round(gpu_weights_mib + gpu_kv_mib),
            "ram_mib_estimate": round(cpu_required_mib),
        }
        return result, []

    return None, [
        "No candidate context fit the selected memory limits. "
        "Reduce the RAM budget, choose a different resource mode, or use a smaller model."
    ]


def benchmark(
    settings: dict[str, Any],
    config_path: Path,
    recommendation: dict[str, Any],
) -> float:
    """Load the recommended model settings and measure short generation speed."""
    from llama_cpp import Llama

    server = settings["server"]
    model_path = Path(server["model_path"]).expanduser()
    if not model_path.is_absolute():
        model_path = config_path.parent / model_path
    kv_type = server["kv_type"]
    type_ids = {"f16": 1, "q8_0": 8, "q4_0": 2}
    model = Llama(
        model_path=str(model_path),
        n_gpu_layers=recommendation["n_gpu_layers"],
        n_ctx=recommendation["n_ctx"],
        n_batch=server["n_batch"],
        n_ubatch=server["n_ubatch"],
        n_threads=recommendation["n_threads"],
        n_threads_batch=recommendation["n_threads_batch"],
        type_k=type_ids[kv_type],
        type_v=type_ids[kv_type],
        flash_attn=kv_type != "f16",
        logits_all=False,
        chat_format=server["chat_format"],
        verbose=False,
    )
    try:
        started = time.perf_counter()
        response = model.create_completion(
            "Write one short sentence about Proxmox.",
            max_tokens=24,
            temperature=0,
        )
        elapsed = time.perf_counter() - started
        tokens = response["usage"]["completion_tokens"]
        if elapsed <= 0 or tokens <= 0:
            raise RuntimeError("benchmark returned no generated tokens")
        return tokens / elapsed
    finally:
        model.close()


def tune_settings(
    settings: dict[str, Any],
    config_path: Path,
    *,
    input_fn: Input,
    report: Report,
) -> dict[str, Any] | None:
    model_path = Path(settings["server"]["model_path"]).expanduser()
    if not model_path.is_absolute():
        model_path = config_path.parent / model_path
    if not model_path.is_file():
        report(f"Model file does not exist: {model_path}")
        return None

    cpu_count = os.cpu_count() or 1
    available_ram, total_ram = detect_memory()
    gpu = detect_gpu()
    model_size_mib = model_path.stat().st_size / MIB

    report(f"Model: {model_path} ({model_size_mib:,.0f} MiB)")
    report(f"CPU: {cpu_count} logical cores")
    report(f"RAM: {available_ram:,} MiB available / {total_ram:,} MiB total")
    if gpu:
        report(
            f"GPU: {gpu['name']} — {gpu['free_mib']:,} MiB free "
            f"/ {gpu['total_mib']:,} MiB total"
        )
        resource_mode = _ask_choice(
            "Resource preference: GPU-only, GPU with CPU/RAM offload, or CPU-only?",
            ("gpu-only", "gpu+cpu", "cpu-only"),
            "gpu+cpu",
            input_fn=input_fn,
            report=report,
        )
    else:
        report("GPU: no NVIDIA GPU detected; this tuner can only assess CPU/RAM.")
        resource_mode = "cpu-only"

    if resource_mode == "gpu-only" and gpu is None:
        report("Cannot choose GPU-only mode without a detected NVIDIA GPU.")
        return None

    report(
        "CPU/RAM offload means model layers may use system memory. "
        "Even GPU-only inference still uses some CPU and RAM for the process."
    )
    default_ram_budget = max(MODEL_LOAD_RESERVE_MIB, available_ram // 2)
    if resource_mode in {"gpu+cpu", "cpu-only"}:
        ram_budget = _ask_int(
            "Maximum additional RAM the model may use (MiB)",
            default_ram_budget,
            MODEL_LOAD_RESERVE_MIB,
            max(MODEL_LOAD_RESERVE_MIB, available_ram),
            input_fn=input_fn,
            report=report,
        )
    else:
        ram_budget = MODEL_LOAD_RESERVE_MIB

    thread_share = int(
        _ask_choice(
            "Maximum CPU thread share for inference?",
            ("25", "50", "75", "100"),
            "50",
            input_fn=input_fn,
            report=report,
        )
    )
    cpu_threads = max(1, math.ceil(cpu_count * thread_share / 100))
    scan_metadata = _ask_choice(
        "Read GGUF vocabulary metadata only? No model weights will be loaded.",
        ("yes", "no"),
        "yes",
        input_fn=input_fn,
        report=report,
    )
    if scan_metadata == "no":
        report("Cannot estimate context memory without model architecture metadata.")
        return None

    try:
        metadata = scan_model_metadata(model_path)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        report(f"Could not read model metadata: {exc}")
        return None

    recommendation, warnings = choose_recommendation(
        settings,
        model_size_mib=model_size_mib,
        metadata=metadata,
        gpu=gpu,
        ram_budget_mib=ram_budget,
        resource_mode=resource_mode,
        cpu_threads=cpu_threads,
    )
    if recommendation is None:
        for warning in warnings:
            report(f"Unable to recommend settings: {warning}")
        return None

    report("Hardware-based recommendation (estimates, not a benchmark):")
    report(f"  GPU layers: {recommendation['n_gpu_layers']}")
    report(f"  Context:    {recommendation['n_ctx']:,} tokens")
    report(f"  CPU threads:{recommendation['n_threads']:>4}")
    report(f"  KV cache:   ~{recommendation['kv_mib_estimate']:,} MiB")
    report(f"  GPU memory: ~{recommendation['gpu_mib_estimate']:,} MiB")
    report(f"  System RAM: ~{recommendation['ram_mib_estimate']:,} MiB")

    if _ask_choice(
        "Load the model and run a short throughput test? This can use substantial resources.",
        ("yes", "no"),
        "no",
        input_fn=input_fn,
        report=report,
    ) == "yes":
        if _ask_choice(
            "Confirm you are ready for the model to load now?",
            ("yes", "no"),
            "no",
            input_fn=input_fn,
            report=report,
        ) != "yes":
            report("Benchmark skipped.")
        else:
            try:
                tokens_per_second = benchmark(settings, config_path, recommendation)
            except KeyboardInterrupt:
                report("Benchmark interrupted; config has not been changed.")
                return None
            except (OSError, RuntimeError, ValueError) as exc:
                report(f"Benchmark failed: {exc}")
                return None
            report(f"Measured generation speed: {tokens_per_second:.2f} tokens/second")

    updated = {section: options.copy() for section, options in settings.items()}
    updated_server = updated["server"]
    updated_server["n_gpu_layers"] = recommendation["n_gpu_layers"]
    updated_server["n_ctx"] = recommendation["n_ctx"]
    updated_server["n_threads"] = recommendation["n_threads"]
    updated_server["n_threads_batch"] = recommendation["n_threads_batch"]

    return updated
