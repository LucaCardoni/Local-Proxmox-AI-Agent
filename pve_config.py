"""Load shared settings for the Proxmox chat client, agent, and model server."""
from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("config.toml")
EXAMPLE_CONFIG_PATH = Path(__file__).resolve().with_name("config.example.toml")

SCHEMA: dict[str, dict[str, type]] = {
    "llm": {
        "base_url": str,
        "model": str,
        "temperature": float,
        "request_timeout": int,
    },
    "ssh": {
        "host": str,
        "user": str,
        "timeout": int,
        "connect_timeout": int,
        "batch_mode": bool,
        "strict_host_key_checking": str,
    },
    "agent": {
        "max_steps": int,
        "max_tokens": int,
        "auto_confirm_safe": bool,
    },
    "chat": {
        "max_tokens": int,
        "context_tokens": int,
        "context_reserve": int,
        "context_detection_timeout": int,
        "max_tool_chars": int,
        "memory_min_tokens": int,
        "memory_max_tokens": int,
    },
    "server": {
        "model_path": str,
        "model_alias": str,
        "host": str,
        "port": int,
        "n_gpu_layers": int,
        "n_ctx": int,
        "n_batch": int,
        "n_ubatch": int,
        "n_threads": int,
        "n_threads_batch": int,
        "kv_type": str,
        "chat_format": str,
        "model_vram_mib": int,
        "embedding_size_estimate": int,
    },
}

ENV_OVERRIDES = {
    "PVE_AGENT_BASE_URL": ("llm", "base_url", str),
    "PVE_AGENT_MODEL": ("llm", "model", str),
    "PVE_AGENT_TEMPERATURE": ("llm", "temperature", float),
    "PVE_AGENT_SSH_HOST": ("ssh", "host", str),
    "PVE_AGENT_SSH_USER": ("ssh", "user", str),
    "PVE_AGENT_SSH_TIMEOUT": ("ssh", "timeout", int),
    "PVE_AGENT_SSH_CONNECT_TIMEOUT": ("ssh", "connect_timeout", int),
    "PVE_AGENT_MAX_STEPS": ("agent", "max_steps", int),
    "PVE_AGENT_MAX_TOKENS": ("agent", "max_tokens", int),
    "PVE_CHAT_MAX_TOKENS": ("chat", "max_tokens", int),
    "PVE_CHAT_CTX_TOKENS": ("chat", "context_tokens", int),
    "PVE_CHAT_CTX_RESERVE": ("chat", "context_reserve", int),
    "PVE_CHAT_MAX_TOOL_CHARS": ("chat", "max_tool_chars", int),
    "PVE_N_CTX": ("server", "n_ctx", int),
    "PVE_KV_TYPE": ("server", "kv_type", str),
}


def _validate_settings(values: dict[str, Any]) -> None:
    for section, options in values.items():
        if section not in SCHEMA or not isinstance(options, dict):
            raise ValueError(f"unknown or invalid configuration section: {section}")
    for section, schema in SCHEMA.items():
        options = values.get(section)
        if not isinstance(options, dict):
            raise ValueError(f"missing configuration section: [{section}]")
        for key, expected in schema.items():
            if key not in options:
                raise ValueError(f"missing configuration setting: {section}.{key}")
            value = options[key]
            valid = (
                isinstance(value, expected)
                if expected is not int
                else isinstance(value, int) and not isinstance(value, bool)
            )
            if not valid:
                raise ValueError(
                    f"invalid value for {section}.{key}: expected {expected.__name__}"
                )
        unknown = options.keys() - schema.keys()
        if unknown:
            raise ValueError(
                f"unknown configuration setting: {section}.{sorted(unknown)[0]}"
            )


def load_settings(
    config_path: str | Path | None = None,
    *,
    apply_env_overrides: bool = True,
) -> tuple[dict[str, Any], Path]:
    """Load TOML settings and legacy environment overrides.

    Relative model paths are resolved against the configuration file's directory
    by the caller, so a custom config file is portable with its model files.
    """
    path = Path(config_path).expanduser().resolve() if config_path else DEFAULT_CONFIG_PATH
    try:
        with path.open("rb") as stream:
            values = tomllib.load(stream)
    except OSError as exc:
        raise ValueError(f"cannot read config file {path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"invalid TOML in {path}: {exc}") from exc

    _validate_settings(values)
    if apply_env_overrides:
        for name, (section, key, converter) in ENV_OVERRIDES.items():
            if name in os.environ:
                try:
                    values[section][key] = converter(os.environ[name])
                except ValueError as exc:
                    raise ValueError(f"invalid {name}: {exc}") from exc
    _validate_settings(values)
    return values, path
