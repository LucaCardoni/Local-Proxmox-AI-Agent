#!/usr/bin/env python3
"""Interactive setup wizard for pve-chat's shared config.toml."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

from pve_config import DEFAULT_CONFIG_PATH, load_settings


Input = Callable[[str], str]


def _ask_text(
    label: str,
    current: str,
    *,
    input_fn: Input,
    allow_empty: bool = False,
) -> str:
    while True:
        answer = input_fn(f"{label} [{current}]: ").strip()
        if not answer:
            if allow_empty:
                return ""
            return current
        return answer


def _ask_int(
    label: str,
    current: int,
    *,
    input_fn: Input,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    while True:
        answer = input_fn(f"{label} [{current}]: ").strip()
        if not answer:
            return current
        try:
            value = int(answer)
        except ValueError:
            print("Enter a whole number.")
            continue
        if minimum is not None and value < minimum:
            print(f"Enter a number greater than or equal to {minimum}.")
            continue
        if maximum is not None and value > maximum:
            print(f"Enter a number less than or equal to {maximum}.")
            continue
        return value


def _ask_float(
    label: str,
    current: float,
    *,
    input_fn: Input,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    while True:
        answer = input_fn(f"{label} [{current}]: ").strip()
        if not answer:
            return current
        try:
            value = float(answer)
        except ValueError:
            print("Enter a number.")
            continue
        if (minimum is not None and value < minimum) or (
            maximum is not None and value > maximum
        ):
            bounds = f"{minimum} to {maximum}" if maximum is not None else f"at least {minimum}"
            print(f"Enter a value {bounds}.")
            continue
        return value


def _ask_bool(label: str, current: bool, *, input_fn: Input) -> bool:
    default = "Y/n" if current else "y/N"
    while True:
        answer = input_fn(f"{label} ({default}): ").strip().casefold()
        if not answer:
            return current
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Enter yes or no.")


def _ask_choice(
    label: str,
    current: str,
    choices: tuple[str, ...],
    *,
    input_fn: Input,
) -> str:
    options = "/".join(choices)
    while True:
        answer = input_fn(f"{label} ({options}) [{current}]: ").strip()
        if not answer:
            if current in choices:
                return current
            print(f"Choose one of: {options}.")
            continue
        if answer in choices:
            return answer
        print(f"Choose one of: {options}.")


def configure(settings: dict[str, Any], *, input_fn: Input = input) -> dict[str, Any]:
    """Prompt for commonly changed values, optionally including advanced settings."""
    result = json.loads(json.dumps(settings))
    llm, ssh = result["llm"], result["ssh"]
    agent, chat, server = (
        result["agent"], result["chat"], result["server"]
    )

    print("\nBasic settings (press Enter to keep the value in brackets).")
    print("\nLLM connection")
    llm["base_url"] = _ask_text("OpenAI-compatible API URL", llm["base_url"], input_fn=input_fn)
    llm["model"] = _ask_text("API model name", llm["model"], input_fn=input_fn)
    llm["temperature"] = _ask_float(
        "Temperature (0-2)", llm["temperature"],
        input_fn=input_fn, minimum=0, maximum=2,
    )

    print("\nProxmox SSH")
    ssh["host"] = _ask_text("SSH host or alias", ssh["host"], input_fn=input_fn)
    ssh["user"] = _ask_text("SSH username", ssh["user"], input_fn=input_fn)
    print("Host-key checking protects against connecting to an untrusted server.")
    ssh["strict_host_key_checking"] = _ask_choice(
        "SSH host-key checking", ssh["strict_host_key_checking"],
        ("yes", "accept-new", "no"), input_fn=input_fn,
    )

    print("\nLocal model server")
    server["model_path"] = _ask_text(
        "GGUF model path (relative to config.toml or absolute)",
        server["model_path"], input_fn=input_fn,
    )
    server["host"] = _ask_text("Listen address", server["host"], input_fn=input_fn)
    server["port"] = _ask_int(
        "Listen port", server["port"], input_fn=input_fn,
        minimum=1, maximum=65535,
    )
    server["n_ctx"] = _ask_int(
        "Context size", server["n_ctx"], input_fn=input_fn, minimum=1024
    )
    server["n_gpu_layers"] = _ask_int(
        "GPU layers (-1 uses all available layers)",
        server["n_gpu_layers"], input_fn=input_fn, minimum=-1,
    )
    server["kv_type"] = _ask_choice(
        "KV cache type", server["kv_type"], ("f16", "q8_0", "q4_0"),
        input_fn=input_fn,
    )

    print("\nSafety")
    print("Safe-classified commands can run without confirmation when enabled.")
    agent["auto_confirm_safe"] = _ask_bool(
        "Automatically confirm safe commands", agent["auto_confirm_safe"],
        input_fn=input_fn,
    )

    if _ask_bool("Configure advanced settings too?", False, input_fn=input_fn):
        print("\nLLM and SSH timeouts")
        llm["request_timeout"] = _ask_int(
            "LLM request timeout (seconds)", llm["request_timeout"],
            input_fn=input_fn, minimum=1,
        )
        ssh["timeout"] = _ask_int(
            "SSH command timeout (seconds)", ssh["timeout"],
            input_fn=input_fn, minimum=1,
        )
        ssh["connect_timeout"] = _ask_int(
            "SSH connection timeout (seconds)", ssh["connect_timeout"],
            input_fn=input_fn, minimum=1,
        )
        ssh["batch_mode"] = _ask_bool(
            "Use SSH batch mode (no password prompts)", ssh["batch_mode"],
            input_fn=input_fn,
        )

        print("\nAgent and chat limits")
        agent["max_steps"] = _ask_int(
            "Maximum agent steps per request", agent["max_steps"],
            input_fn=input_fn, minimum=1,
        )
        agent["max_tokens"] = _ask_int(
            "Agent response token limit", agent["max_tokens"],
            input_fn=input_fn, minimum=1,
        )
        chat["max_tokens"] = _ask_int(
            "Chat response token limit", chat["max_tokens"],
            input_fn=input_fn, minimum=1,
        )
        chat["context_tokens"] = _ask_int(
            "Fallback context size", chat["context_tokens"],
            input_fn=input_fn, minimum=1024,
        )
        chat["context_reserve"] = _ask_int(
            "Context reserve", chat["context_reserve"],
            input_fn=input_fn, minimum=0,
        )
        chat["context_detection_timeout"] = _ask_int(
            "Context detection timeout (seconds)",
            chat["context_detection_timeout"], input_fn=input_fn, minimum=1,
        )
        chat["max_tool_chars"] = _ask_int(
            "Maximum tool-result characters", chat["max_tool_chars"],
            input_fn=input_fn, minimum=512,
        )
        chat["memory_min_tokens"] = _ask_int(
            "Minimum memory-update tokens", chat["memory_min_tokens"],
            input_fn=input_fn, minimum=1,
        )
        chat["memory_max_tokens"] = _ask_int(
            "Maximum memory-update tokens", chat["memory_max_tokens"],
            input_fn=input_fn, minimum=1,
        )

        print("\nAdvanced model-server tuning")
        server["model_alias"] = _ask_text(
            "Model alias (use {n_ctx} for context size)",
            server["model_alias"], input_fn=input_fn,
        )
        server["n_batch"] = _ask_int(
            "Batch size", server["n_batch"], input_fn=input_fn, minimum=1
        )
        server["n_ubatch"] = _ask_int(
            "Micro-batch size", server["n_ubatch"], input_fn=input_fn, minimum=1
        )
        server["n_threads"] = _ask_int(
            "Inference CPU threads", server["n_threads"],
            input_fn=input_fn, minimum=1,
        )
        server["n_threads_batch"] = _ask_int(
            "Batch CPU threads", server["n_threads_batch"],
            input_fn=input_fn, minimum=1,
        )
        server["chat_format"] = _ask_text(
            "llama.cpp chat format", server["chat_format"], input_fn=input_fn
        ).strip()
        server["model_vram_mib"] = _ask_int(
            "Estimated model VRAM (MiB)", server["model_vram_mib"],
            input_fn=input_fn, minimum=0,
        )
        server["embedding_size_estimate"] = _ask_int(
            "Estimated model embedding size",
            server["embedding_size_estimate"], input_fn=input_fn, minimum=1,
        )

    if chat["memory_min_tokens"] > chat["memory_max_tokens"]:
        raise ValueError("memory_min_tokens must not exceed memory_max_tokens")
    return result


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (int, float)):
        return repr(value)
    raise ValueError(f"cannot write TOML value of type {type(value).__name__}")


def serialize_settings(settings: dict[str, Any]) -> str:
    lines = [
        "# Generated by setup_config.py. You can edit this file directly.",
        "# Relative model paths are resolved from this file's directory.",
        "",
    ]
    for section, options in settings.items():
        lines.append(f"[{section}]")
        lines.extend(f"{key} = {_toml_value(value)}" for key, value in options.items())
        lines.append("")
    return "\n".join(lines)


def write_config(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="config file to create or update (default: config.toml)",
    )
    args = parser.parse_args(argv)
    config_path = args.config.expanduser().resolve()

    try:
        settings, _ = load_settings(config_path, apply_env_overrides=False)
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Config file: {config_path}")
    print("No SSH or model-server changes are made by this setup wizard.")
    try:
        updated = configure(settings)
        content = serialize_settings(updated)
        # Validate the complete TOML before changing the user's config.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=config_path.parent,
            prefix=f".{config_path.name}.",
            suffix=".check",
            delete=False,
        ) as stream:
            check_path = Path(stream.name)
            stream.write(content)
        try:
            load_settings(check_path, apply_env_overrides=False)
        finally:
            check_path.unlink(missing_ok=True)

        if not _ask_bool(
            f"Save these settings to {config_path}?",
            True,
            input_fn=input,
        ):
            print("No changes saved.")
            return 0
        write_config(config_path, content)
    except (ValueError, OSError, EOFError, KeyboardInterrupt) as exc:
        if isinstance(exc, KeyboardInterrupt):
            print("\nSetup cancelled; config was not changed.")
            return 130
        if isinstance(exc, EOFError):
            print("\nInput ended; config was not changed.")
            return 1
        print(f"Setup failed: {exc}")
        return 1

    model_path = Path(updated["server"]["model_path"]).expanduser()
    if not model_path.is_absolute():
        model_path = config_path.parent / model_path
    print(f"Saved settings to {config_path}")
    if not model_path.is_file():
        print(f"Note: model file not found yet: {model_path}")
    print("Start the model server with: python3 serve.py")
    print("Start the chat client with:  python3 pve-chat.py")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
