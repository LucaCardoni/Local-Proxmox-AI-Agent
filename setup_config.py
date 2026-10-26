#!/usr/bin/env python3
"""Interactive setup wizard for pve-chat's shared config.toml."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from pve_config import DEFAULT_CONFIG_PATH, EXAMPLE_CONFIG_PATH, load_settings
from hardware_tuning import tune_settings


Input = Callable[[str], str]
CONSOLE = Console()
ORANGE = os.environ.get("PVE_CHAT_TITLE", "#e56f00")
MUTED = os.environ.get("PVE_CHAT_DIM", "#808080")

BANNER_ART = [
    ("    ████▄     ▄████    ", "    KKKKK     KKKKK    "),
    ("▄▄▄▄▄▀████▄ ▄████▀▄▄▄▄▄", "OOOOOKKKKKK KKKKKKOOOOO"),
    (" ▀████▄▀███████▀▄████▀ ", " OOOOOOKKKKKKKKKOOOOOO "),
    ("   ▀████▄▀███▀▄████▀   ", "   OOOOOOKKKKKOOOOOO   "),
    ("     ████▄ ▀ ▄████▀    ", "     OOOOO K OOOOOO    "),
    ("    ▄████▀ ▄ ▀████▄    ", "    OOOOOO K OOOOOO    "),
    ("   ▄████▀▄███▄ ████▄   ", "   OOOOOOKKKKK OOOOO   "),
    (" ▄████▀▄███████▄▀████▄ ", " OOOOOOKKKKKKKKKOOOOOO "),
    ("▀▀▀▀▀▄████▀ ▀████▄▀▀▀▀▀", "OOOOOKKKKKK KKKKKKOOOOO"),
    ("    ▀███▀     ▀███▀    ", "    KKKKK     KKKKK    "),
]


def _input(prompt: str) -> str:
    return CONSOLE.input(Text("❯ ", style=f"bold {ORANGE}") + Text(prompt))


def _say(message: str, *, style: str | None = None) -> None:
    CONSOLE.print(Text(message, style=style))


def _section(title: str, description: str = "") -> None:
    body = Text(title, style=f"bold {ORANGE}")
    if description:
        body.append(f"\n{description}", style=MUTED)
    CONSOLE.print(Panel.fit(body, border_style=ORANGE, padding=(0, 1)))


def _banner(config_path: Path) -> None:
    if CONSOLE.is_terminal:
        CONSOLE.clear()
    info = [
        ("PVE-Chat Setup", f"bold {ORANGE}"),
        ("Configure your local Proxmox AI assistant", MUTED),
        (str(config_path), MUTED),
        ("Enter keeps the current value", MUTED),
        ("Ctrl+C cancels without saving", MUTED),
    ]
    top = max(0, (len(BANNER_ART) - len(info)) // 2)
    for row_index, (art, colors) in enumerate(BANNER_ART):
        row = Text()
        run = ""
        run_style: str | None = None
        for char, color in zip(art, colors):
            style = (
                f"bold {ORANGE}" if color == "O"
                else f"bold {MUTED}" if color == "K"
                else None
            )
            if run and style != run_style:
                row.append(run, style=run_style)
                run = ""
            run += char
            run_style = style
        if run:
            row.append(run, style=run_style)
        if top <= row_index < top + len(info):
            label, style = info[row_index - top]
            row.append("   ")
            row.append(label, style=style)
        CONSOLE.print(row)
    CONSOLE.print()


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
            _say("Enter a whole number.", style="yellow")
            continue
        if minimum is not None and value < minimum:
            _say(f"Enter a number greater than or equal to {minimum}.", style="yellow")
            continue
        if maximum is not None and value > maximum:
            _say(f"Enter a number less than or equal to {maximum}.", style="yellow")
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
            _say("Enter a number.", style="yellow")
            continue
        if (minimum is not None and value < minimum) or (
            maximum is not None and value > maximum
        ):
            bounds = f"{minimum} to {maximum}" if maximum is not None else f"at least {minimum}"
            _say(f"Enter a value {bounds}.", style="yellow")
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
        _say("Enter yes or no.", style="yellow")


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
            _say(f"Choose one of: {options}.", style="yellow")
            continue
        if answer in choices:
            return answer
        _say(f"Choose one of: {options}.", style="yellow")


def configure(
    settings: dict[str, Any],
    config_path: Path,
    *,
    input_fn: Input = _input,
) -> dict[str, Any]:
    """Prompt for commonly changed values, optionally including advanced settings."""
    result = json.loads(json.dumps(settings))
    llm, ssh = result["llm"], result["ssh"]
    agent, chat, server = (
        result["agent"], result["chat"], result["server"]
    )

    _section("LLM connection")
    llm["base_url"] = _ask_text("OpenAI-compatible API URL", llm["base_url"], input_fn=input_fn)
    llm["model"] = _ask_text("API model name", llm["model"], input_fn=input_fn)
    llm["temperature"] = _ask_float(
        "Temperature (0-2)", llm["temperature"],
        input_fn=input_fn, minimum=0, maximum=2,
    )

    _section("Proxmox SSH")
    ssh["host"] = _ask_text("SSH host or alias", ssh["host"], input_fn=input_fn)
    ssh["user"] = _ask_text("SSH username", ssh["user"], input_fn=input_fn)
    _say("Host-key checking protects against connecting to an untrusted server.", style=MUTED)
    ssh["strict_host_key_checking"] = _ask_choice(
        "SSH host-key checking", ssh["strict_host_key_checking"],
        ("yes", "accept-new", "no"), input_fn=input_fn,
    )

    _section("Local model server")
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

    model_path = Path(server["model_path"]).expanduser()
    if not model_path.is_absolute():
        model_path = config_path.parent / model_path
    if _ask_bool(
        "Find a suitable model configuration from this machine's hardware?",
        model_path.is_file(),
        input_fn=input_fn,
    ):
        _section(
            "Hardware tuner",
            "Estimates memory fit and can benchmark model settings for speed.",
        )

        def report_tuning(message: str) -> None:
            style = (
                f"bold {ORANGE}" if message.startswith("Hardware-based")
                else "yellow" if message.startswith(
                    ("Unable to", "Could not", "Model file", "Benchmark failed")
                )
                else MUTED
            )
            _say(message, style=style)

        tuned = tune_settings(
            result,
            config_path,
            input_fn=input_fn,
            report=report_tuning,
        )
        if tuned is not None:
            result = tuned
            llm, ssh = result["llm"], result["ssh"]
            agent, chat, server = (
                result["agent"], result["chat"], result["server"]
            )

    _section("Safety")
    _say("Safe-classified commands can run without confirmation when enabled.", style=MUTED)
    agent["auto_confirm_safe"] = _ask_bool(
        "Automatically confirm safe commands", agent["auto_confirm_safe"],
        input_fn=input_fn,
    )

    if _ask_bool("Configure advanced settings too?", False, input_fn=input_fn):
        _section("LLM and SSH timeouts")
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

        _section("Agent and chat limits")
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

        _section("Advanced model-server settings")
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
        seed_path = config_path if config_path.is_file() else EXAMPLE_CONFIG_PATH
        settings, _ = load_settings(seed_path, apply_env_overrides=False)
    except ValueError as exc:
        parser.error(str(exc))

    _banner(config_path)
    _say("No SSH or model-server changes are made by this setup wizard.", style=MUTED)
    try:
        updated = configure(settings, config_path)
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
            input_fn=_input,
        ):
            _say("No changes saved.", style=MUTED)
            return 0
        write_config(config_path, content)
    except (ValueError, OSError, EOFError, KeyboardInterrupt) as exc:
        if isinstance(exc, KeyboardInterrupt):
            _say("\nSetup cancelled; config was not changed.", style="yellow")
            return 130
        if isinstance(exc, EOFError):
            _say("\nInput ended; config was not changed.", style="yellow")
            return 1
        _say(f"Setup failed: {exc}", style="bold red")
        return 1

    model_path = Path(updated["server"]["model_path"]).expanduser()
    if not model_path.is_absolute():
        model_path = config_path.parent / model_path
    _say(f"Saved settings to {config_path}", style=f"bold {ORANGE}")
    if not model_path.is_file():
        _say(f"Note: model file not found yet: {model_path}", style="yellow")
    _say("Run the launcher with: python3 main.py", style=MUTED)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
