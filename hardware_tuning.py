#!/usr/bin/env python3
"""Hardware detection and model-server recommendations for the setup wizard."""
from __future__ import annotations

import ctypes
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Callable


KV_BYTES = {"f16": 2.0, "q8_0": 1.0, "q4_0": 0.5}
MIB = 1024 * 1024
MODEL_LOAD_RESERVE_MIB = 256
DXGI_ADAPTER1_GET_DESC1_INDEX = 10
DXGI_ADAPTER3_QUERY_VIDEO_MEMORY_INFO_INDEX = 14
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

    return (
        _detect_windows_memory()
        if os.name == "nt"
        else _detect_posix_memory()
    )


def _detect_posix_memory() -> tuple[int, int]:
    page_size = os.sysconf("SC_PAGE_SIZE")
    total_pages = os.sysconf("SC_PHYS_PAGES")
    available_pages = os.sysconf("SC_AVPHYS_PAGES")
    return available_pages * page_size // MIB, total_pages * page_size // MIB


def _detect_windows_memory() -> tuple[int, int]:
    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_uint32),
            ("dwMemoryLoad", ctypes.c_uint32),
            ("ullTotalPhys", ctypes.c_uint64),
            ("ullAvailPhys", ctypes.c_uint64),
            ("ullTotalPageFile", ctypes.c_uint64),
            ("ullAvailPageFile", ctypes.c_uint64),
            ("ullTotalVirtual", ctypes.c_uint64),
            ("ullAvailVirtual", ctypes.c_uint64),
            ("ullAvailExtendedVirtual", ctypes.c_uint64),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    global_memory_status_ex = ctypes.WinDLL(
        "kernel32", use_last_error=True
    ).GlobalMemoryStatusEx
    global_memory_status_ex.argtypes = [ctypes.POINTER(MemoryStatusEx)]
    global_memory_status_ex.restype = ctypes.c_int
    if not global_memory_status_ex(ctypes.byref(status)):
        raise ctypes.WinError(ctypes.get_last_error())
    return status.ullAvailPhys // MIB, status.ullTotalPhys // MIB


def detect_gpu() -> dict[str, Any] | None:
    """Return the first supported GPU's name and memory usage, if available."""
    nvidia_gpu = _detect_nvidia_gpu()
    if nvidia_gpu is not None:
        return nvidia_gpu
    operating_system = platform.system()
    if operating_system == "Windows":
        return _detect_windows_amd_gpu()
    if operating_system == "Linux":
        return _detect_linux_amd_gpu()
    return None


def _detect_nvidia_gpu() -> dict[str, Any] | None:
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


def _detect_linux_amd_gpu(
    drm_root: Path = Path("/sys/class/drm"),
) -> dict[str, Any] | None:
    for device_path in drm_root.glob("card[0-9]*"):
        if re.fullmatch(r"card\d+", device_path.name) is None:
            continue
        device = device_path / "device"
        try:
            if (device / "vendor").read_text(encoding="ascii").strip().casefold() != "0x1002":
                continue
            total_bytes = int(
                (device / "mem_info_vram_total").read_text(encoding="ascii").strip()
            )
            used_bytes = int(
                (device / "mem_info_vram_used").read_text(encoding="ascii").strip()
            )
        except (OSError, ValueError):
            continue
        if total_bytes <= 0:
            continue

        name = "AMD GPU"
        for name_file in ("product_name", "model_name"):
            try:
                candidate = (device / name_file).read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if candidate:
                name = candidate
                break
        return {
            "name": name,
            "total_mib": total_bytes // MIB,
            "free_mib": max(0, total_bytes - used_bytes) // MIB,
        }
    return None


def _detect_windows_amd_gpu() -> dict[str, Any] | None:
    class Guid(ctypes.Structure):
        _fields_ = [
            ("data1", ctypes.c_uint32),
            ("data2", ctypes.c_uint16),
            ("data3", ctypes.c_uint16),
            ("data4", ctypes.c_ubyte * 8),
        ]

    class AdapterDescription(ctypes.Structure):
        _fields_ = [
            ("description", ctypes.c_wchar * 128),
            ("vendor_id", ctypes.c_uint32),
            ("device_id", ctypes.c_uint32),
            ("subsystem_id", ctypes.c_uint32),
            ("revision", ctypes.c_uint32),
            ("dedicated_video_memory", ctypes.c_size_t),
            ("dedicated_system_memory", ctypes.c_size_t),
            ("shared_system_memory", ctypes.c_size_t),
            ("adapter_luid", ctypes.c_byte * 8),
            ("flags", ctypes.c_uint32),
        ]

    class VideoMemoryInfo(ctypes.Structure):
        _fields_ = [
            ("budget", ctypes.c_uint64),
            ("current_usage", ctypes.c_uint64),
            ("current_reservation", ctypes.c_uint64),
            ("available_for_reservation", ctypes.c_uint64),
        ]

    factory_iid = Guid(
        0x770AAE78, 0xF26F, 0x4DBA,
        (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87),
    )
    adapter3_iid = Guid(
        0x645967A4, 0x1392, 0x4310,
        (ctypes.c_ubyte * 8)(0xA7, 0x98, 0x80, 0x53, 0xCE, 0x3E, 0x93, 0xFD),
    )
    hresult = ctypes.c_long
    winfun = ctypes.WINFUNCTYPE

    def com_method(interface: ctypes.c_void_p, index: int, signature: Any) -> Any:
        vtable = ctypes.cast(
            interface, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
        ).contents
        return signature(vtable[index])

    def release(interface: ctypes.c_void_p) -> None:
        if interface:
            com_method(interface, 2, winfun(hresult, ctypes.c_void_p))(interface)

    factory = ctypes.c_void_p()
    create_factory = ctypes.WinDLL("dxgi").CreateDXGIFactory1
    create_factory.argtypes = [ctypes.POINTER(Guid), ctypes.POINTER(ctypes.c_void_p)]
    create_factory.restype = hresult
    if create_factory(ctypes.byref(factory_iid), ctypes.byref(factory)) < 0:
        return None

    try:
        enum_adapter = com_method(
            factory,
            12,
            winfun(hresult, ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)),
        )
        for index in range(32):
            adapter = ctypes.c_void_p()
            result = enum_adapter(factory, index, ctypes.byref(adapter))
            if result < 0 or not adapter:
                break
            try:
                description = AdapterDescription()
                get_description = com_method(
                    adapter,
                    DXGI_ADAPTER1_GET_DESC1_INDEX,
                    winfun(hresult, ctypes.c_void_p, ctypes.POINTER(AdapterDescription)),
                )
                if get_description(adapter, ctypes.byref(description)) < 0:
                    continue
                if description.vendor_id != 0x1002 or description.dedicated_video_memory <= 0:
                    continue

                adapter3 = ctypes.c_void_p()
                query_interface = com_method(
                    adapter,
                    0,
                    winfun(
                        hresult,
                        ctypes.c_void_p,
                        ctypes.POINTER(Guid),
                        ctypes.POINTER(ctypes.c_void_p),
                    ),
                )
                if query_interface(adapter, ctypes.byref(adapter3_iid), ctypes.byref(adapter3)) < 0:
                    continue
                try:
                    memory_info = VideoMemoryInfo()
                    query_memory = com_method(
                        adapter3,
                        DXGI_ADAPTER3_QUERY_VIDEO_MEMORY_INFO_INDEX,
                        winfun(
                            hresult,
                            ctypes.c_void_p,
                            ctypes.c_uint,
                            ctypes.c_int,
                            ctypes.POINTER(VideoMemoryInfo),
                        ),
                    )
                    if query_memory(adapter3, 0, 0, ctypes.byref(memory_info)) < 0:
                        continue
                    total_bytes = description.dedicated_video_memory
                    available_bytes = max(
                        0, memory_info.budget - memory_info.current_usage
                    )
                    return {
                        "name": description.description,
                        "total_mib": total_bytes // MIB,
                        "free_mib": available_bytes // MIB,
                    }
                finally:
                    release(adapter3)
            finally:
                release(adapter)
    finally:
        release(factory)
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


def performance_candidates(
    *,
    layers: int,
    max_cpu_threads: int,
    batch_threads: int,
    resource_mode: str,
    gpu_offload_supported: bool,
) -> list[dict[str, int]]:
    """Build a small, quality-preserving GPU-layer and CPU-thread search."""
    if resource_mode == "cpu-only" or not gpu_offload_supported:
        layer_counts = [0]
    elif resource_mode == "gpu-only":
        layer_counts = [layers]
    else:
        layer_counts = [0, max(1, layers // 2), layers]

    thread_counts = sorted(
        {
            count
            for count in (1, 2, 4, 8, max_cpu_threads)
            if 1 <= count <= max_cpu_threads
        }
    )
    return [
        {
            "n_gpu_layers": -1 if layer_count == layers else layer_count,
            "n_threads": threads,
            "n_threads_batch": batch_threads,
        }
        for layer_count in layer_counts
        for threads in thread_counts
    ]


def benchmark_thread_counts(
    settings: dict[str, Any],
    config_path: Path,
    recommendation: dict[str, Any],
    thread_counts: list[int],
    *,
    report: Report,
) -> dict[int, float]:
    """Compare chat decode speed using the same template and JSON prompt as PVE-Chat."""
    from llama_cpp import Llama, llama_chat_format

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
        chat_format=(
            None if server["chat_format"] == "auto" else server["chat_format"]
        ),
        verbose=False,
    )
    try:
        base_chat_handler = (
            model.chat_handler
            or model._chat_handlers.get(model.chat_format)
            or llama_chat_format.get_chat_completion_handler(model.chat_format)
        )

        def chat_handler_with_kwargs(*args: Any, **kwargs: Any) -> Any:
            return base_chat_handler(
                *args,
                **{**kwargs, "enable_thinking": False},
            )

        model.chat_handler = chat_handler_with_kwargs
        rates: dict[int, float] = {}
        messages = [{
            "role": "user",
            "content": (
                "Return one JSON object with an answer containing a concise "
                "technical explanation of why memory bandwidth matters for "
                "local inference."
            ),
        }]
        for threads in thread_counts:
            report(
                f"  Testing {recommendation['n_gpu_layers']} GPU layers "
                f"with {threads} CPU threads..."
            )
            model._ctx.set_n_threads(
                threads, recommendation["n_threads_batch"]
            )
            model.create_chat_completion(
                messages=messages,
                max_tokens=16,
                temperature=0,
            )

            measurements: list[float] = []
            for _ in range(3):
                started: float | None = None
                last_token_at: float | None = None
                token_count = 0
                for chunk in model.create_chat_completion(
                    messages=messages,
                    max_tokens=128,
                    temperature=0,
                    stream=True,
                ):
                    choice = chunk["choices"][0]
                    content = choice.get("delta", {}).get("content")
                    if content:
                        now = time.perf_counter()
                        if started is None:
                            started = now
                        last_token_at = now
                        token_count += 1
                if started is None or last_token_at is None or token_count < 16:
                    raise RuntimeError(
                        "benchmark generated too few tokens to measure decode speed"
                    )
                elapsed = last_token_at - started
                if elapsed <= 0:
                    raise RuntimeError("benchmark returned an invalid duration")
                measurements.append((token_count - 1) / elapsed)
            rates[threads] = statistics.median(measurements)
        return rates
    finally:
        model.close()


def find_fastest_configuration(
    settings: dict[str, Any],
    config_path: Path,
    recommendation: dict[str, Any],
    *,
    metadata: dict[str, int | str],
    gpu: dict[str, Any] | None,
    model_size_mib: float,
    ram_budget_mib: int,
    resource_mode: str,
    max_cpu_threads: int,
    report: Report,
) -> dict[str, int] | None:
    """Benchmark feasible offload/thread combinations without lowering context."""
    server = settings["server"]
    context = int(server["n_ctx"])
    if context > int(metadata["context_length"]):
        report(
            "Cannot speed-test the current context because it exceeds "
            "the model's supported context."
        )
        return None

    gpu_offload_supported = False
    if gpu is not None:
        try:
            from llama_cpp import llama_cpp
        except ImportError as exc:
            report(f"Could not check GPU offload support: {exc}")
        else:
            gpu_offload_supported = bool(llama_cpp.llama_supports_gpu_offload())
        if not gpu_offload_supported:
            report(
                "The installed llama.cpp backend does not support GPU offload; "
                "the speed search will test CPU-only settings."
            )
            if resource_mode == "gpu-only":
                return None

    candidates = performance_candidates(
        layers=int(metadata["layers"]),
        max_cpu_threads=max_cpu_threads,
        batch_threads=int(server["n_threads_batch"]),
        resource_mode=resource_mode,
        gpu_offload_supported=gpu_offload_supported,
    )
    kv_mib = (
        int(metadata["layers"])
        * int(metadata["kv_heads"])
        * (int(metadata["key_length"]) + int(metadata["value_length"]))
        * KV_BYTES[server["kv_type"]]
        * context
        / MIB
    )
    gpu_budget_mib = 0.0
    if gpu is not None:
        reserve = max(256, gpu["total_mib"] * 0.10)
        gpu_budget_mib = max(0.0, gpu["free_mib"] - reserve)

    feasible: dict[int, list[int]] = {}
    layer_count = int(metadata["layers"])
    for candidate in candidates:
        gpu_layers = candidate["n_gpu_layers"]
        if gpu_layers < 0:
            gpu_layers = layer_count
        gpu_fraction = gpu_layers / layer_count
        gpu_required_mib = model_size_mib * 1.10 * gpu_fraction + kv_mib * gpu_fraction
        cpu_required_mib = (
            model_size_mib * (1 - gpu_fraction)
            + kv_mib * (1 - gpu_fraction)
            + MODEL_LOAD_RESERVE_MIB
        )
        if gpu_required_mib > gpu_budget_mib or cpu_required_mib > ram_budget_mib:
            continue
        feasible.setdefault(gpu_layers, []).append(candidate["n_threads"])

    if not feasible:
        report(
            "No speed-test configuration fits the selected GPU/RAM budgets "
            f"at the current {context:,}-token context. Context and KV type "
            "were left unchanged."
        )
        return None

    report(
        "Speed search keeps the model, "
        f"{context:,}-token context, and {server['kv_type']} KV cache unchanged."
    )
    results: list[tuple[float, int, int]] = []
    for gpu_layers, thread_counts in feasible.items():
        candidate = recommendation.copy()
        candidate["n_ctx"] = context
        candidate["n_gpu_layers"] = (
            -1 if gpu_layers == layer_count else gpu_layers
        )
        try:
            rates = benchmark_thread_counts(
                settings,
                config_path,
                candidate,
                thread_counts,
                report=report,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            report(
                f"Speed test failed for {gpu_layers} GPU layers: {exc}"
            )
            continue
        for threads, tokens_per_second in rates.items():
            report(
                f"  {gpu_layers} GPU layers, {threads} CPU threads: "
                f"{tokens_per_second:.2f} generated tokens/second"
            )
            results.append((tokens_per_second, gpu_layers, threads))

    if not results:
        report("No speed-test configuration completed successfully.")
        return None

    tokens_per_second, gpu_layers, threads = max(
        results, key=lambda result: (result[0], result[1], -result[2])
    )
    report(
        "Fastest measured configuration: "
        f"{gpu_layers} GPU layers, {threads} CPU threads "
        f"({tokens_per_second:.2f} generated tokens/second)."
    )
    return {
        "n_gpu_layers": -1 if gpu_layers == layer_count else gpu_layers,
        "n_threads": threads,
        "n_threads_batch": int(server["n_threads_batch"]),
        "n_ctx": context,
    }


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
        report(
            "GPU memory detection does not verify that the installed "
            "llama.cpp build supports GPU offload."
        )
        resource_mode = _ask_choice(
            "Resource preference: GPU-only, GPU with CPU/RAM offload, or CPU-only?",
            ("gpu-only", "gpu+cpu", "cpu-only"),
            "gpu+cpu",
            input_fn=input_fn,
            report=report,
        )
    else:
        report("GPU: no supported GPU detected; this tuner can only assess CPU/RAM.")
        resource_mode = "cpu-only"

    if resource_mode == "gpu-only" and gpu is None:
        report("Cannot choose GPU-only mode without a detected GPU.")
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
        "Search GPU-layer and CPU-thread settings for the fastest generation? "
        "This loads the model several times and can take a few minutes.",
        ("yes", "no"),
        "yes",
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
                fastest = find_fastest_configuration(
                    settings,
                    config_path,
                    recommendation,
                    metadata=metadata,
                    gpu=gpu,
                    model_size_mib=model_size_mib,
                    ram_budget_mib=ram_budget,
                    resource_mode=resource_mode,
                    max_cpu_threads=cpu_threads,
                    report=report,
                )
            except KeyboardInterrupt:
                report("Speed search interrupted; config has not been changed.")
                return None
            except (OSError, RuntimeError, ValueError) as exc:
                report(f"Speed search failed: {exc}")
                return None
            if fastest is None:
                return None
            recommendation.update(fastest)

    updated = {section: options.copy() for section, options in settings.items()}
    updated_server = updated["server"]
    updated_server["n_gpu_layers"] = recommendation["n_gpu_layers"]
    updated_server["n_ctx"] = recommendation["n_ctx"]
    updated_server["n_threads"] = recommendation["n_threads"]
    updated_server["n_threads_batch"] = recommendation["n_threads_batch"]

    return updated
