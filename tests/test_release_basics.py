from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import main
import setup_config
from hardware_tuning import choose_recommendation
from pve_agent import classify, confirm
from pve_config import EXAMPLE_CONFIG_PATH, load_settings


def test_example_config_loads_and_uses_safe_public_defaults() -> None:
    settings, _ = load_settings(EXAMPLE_CONFIG_PATH, apply_env_overrides=False)

    assert settings["ssh"]["user"] != "root"
    assert settings["ssh"]["strict_host_key_checking"] == "yes"
    assert settings["agent"]["auto_confirm_safe"] is False
    assert settings["server"]["n_ctx"] == 8192


def test_environment_overrides_apply_to_a_local_config(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "config.toml"
    shutil.copyfile(EXAMPLE_CONFIG_PATH, config_path)
    monkeypatch.setenv("PVE_AGENT_SSH_HOST", "pve.internal")

    settings, _ = load_settings(config_path)

    assert settings["ssh"]["host"] == "pve.internal"


def test_setup_bootstraps_missing_config_from_example(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.toml"
    with (
        patch.object(setup_config, "configure", side_effect=lambda settings, _path: settings),
        patch.object(setup_config, "_ask_bool", return_value=True),
    ):
        assert setup_config.main(["--config", str(config_path)]) == 0

    settings, _ = load_settings(config_path, apply_env_overrides=False)
    assert settings["ssh"]["strict_host_key_checking"] == "yes"


def test_command_classifier_flags_dangerous_commands() -> None:
    assert classify("systemctl restart pveproxy") == "dangerous"
    assert classify("pct list") == "safe"


def test_safe_command_requires_approval_by_default(monkeypatch) -> None:
    cfg = SimpleNamespace(
        dry_run=False,
        approval_granted=False,
        confirm=True,
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    assert confirm(cfg, "list containers", "pct list", "safe") is False


def test_yes_approves_safe_command(monkeypatch) -> None:
    cfg = SimpleNamespace(
        dry_run=False,
        approval_granted=False,
        confirm=True,
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")

    assert confirm(cfg, "list containers", "pct list", "safe") is True


def test_auto_confirm_safe_skips_prompt(monkeypatch) -> None:
    cfg = SimpleNamespace(
        dry_run=False,
        approval_granted=False,
        confirm=False,
    )
    input_mock = Mock(side_effect=AssertionError("unexpected prompt"))
    monkeypatch.setattr("builtins.input", input_mock)

    assert confirm(cfg, "list containers", "pct list", "safe") is True
    input_mock.assert_not_called()


def test_dangerous_command_requires_typed_yes(monkeypatch) -> None:
    cfg = SimpleNamespace(
        dry_run=False,
        approval_granted=False,
        confirm=False,
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")

    assert (
        confirm(cfg, "restart service", "systemctl restart pveproxy", "dangerous")
        is False
    )


def test_tuner_returns_cpu_only_recommendation_within_budget() -> None:
    settings, _ = load_settings(EXAMPLE_CONFIG_PATH, apply_env_overrides=False)
    metadata = {
        "layers": 12,
        "context_length": 8192,
        "kv_heads": 2,
        "key_length": 64,
        "value_length": 64,
    }

    recommendation, warnings = choose_recommendation(
        settings,
        model_size_mib=512,
        metadata=metadata,
        gpu=None,
        ram_budget_mib=4096,
        resource_mode="cpu-only",
        cpu_threads=4,
    )

    assert warnings == []
    assert recommendation is not None
    assert recommendation["n_ctx"] == 8192
    assert recommendation["n_gpu_layers"] == 0


def test_launcher_stops_server_after_client_exits() -> None:
    process = Mock()
    settings = {"server": {"host": "127.0.0.1", "port": 8080}}

    with (
        patch.object(main.subprocess, "Popen", return_value=process),
        patch.object(main, "_wait_until_ready", return_value=True),
        patch.object(main.subprocess, "call", return_value=7),
        patch.object(main, "_stop_server") as stop_server,
    ):
        result = main._run_with_client(
            Path("."),
            Path("config.toml"),
            settings,
            30,
            ["python", "client.py"],
            "test client",
        )

    assert result == 7
    stop_server.assert_called_once_with(process)
