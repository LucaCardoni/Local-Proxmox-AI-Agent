from __future__ import annotations

import importlib.util
import io
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import main
import setup_config
from hardware_tuning import choose_recommendation
from pve_agent import classify, confirm, normalize_answer_text
from pve_config import EXAMPLE_CONFIG_PATH, load_settings


CHAT_UI_SPEC = importlib.util.spec_from_file_location(
    "pve_chat_ui", Path(main.__file__).with_name("pve-chat.py")
)
assert CHAT_UI_SPEC is not None and CHAT_UI_SPEC.loader is not None
chat_ui = importlib.util.module_from_spec(CHAT_UI_SPEC)
CHAT_UI_SPEC.loader.exec_module(chat_ui)


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


def test_normalize_answer_text_converts_literal_newline_escapes() -> None:
    assert normalize_answer_text(
        "Memory\\n- Disk\\r\\n- Uptime"
    ) == "Memory\n- Disk\n- Uptime"


def test_chat_answer_panel_normalizes_literal_newlines() -> None:
    with patch.object(chat_ui, "Markdown", side_effect=lambda content: content):
        panel = chat_ui.chat_message_panel(
            "Agent", "Memory\\n- Disk", "agent_fg", "agent_bg"
        )

    assert panel.renderable == "Memory\n- Disk"


def test_restored_tool_output_is_summarized_when_verbose_is_off() -> None:
    result = "container list\n[exit 0]"
    messages = [
        {"role": "assistant", "content": '{"tool":"list_guests","arguments":{}}'},
        {"role": "user", "content": f"[tool list_guests result]\n{result}"},
    ]
    with (
        patch.object(chat_ui, "SHOW_TOOL_OUTPUT", False),
        patch.object(chat_ui, "show_tool_result") as show_result,
        patch.object(chat_ui.console, "print"),
    ):
        chat_ui.show_saved_transcript(messages)

    show_result.assert_called_once_with(
        "list_guests",
        {},
        "pvesh get /cluster/resources --type vm --output-format json",
        "safe",
        result,
    )
    assert chat_ui._last_tool["result"] == result


def test_restored_tool_output_is_shown_when_verbose_is_on() -> None:
    result = "container list\n[exit 0]"
    messages = [
        {"role": "assistant", "content": '{"tool":"list_guests","arguments":{}}'},
        {"role": "user", "content": f"[tool list_guests result]\n{result}"},
    ]
    with (
        patch.object(chat_ui, "SHOW_TOOL_OUTPUT", True),
        patch.object(chat_ui, "show_tool_result") as show_result,
        patch.object(chat_ui.console, "print"),
    ):
        chat_ui.show_saved_transcript(messages)

    show_result.assert_called_once_with(
        "list_guests",
        {},
        "pvesh get /cluster/resources --type vm --output-format json",
        "safe",
        result,
    )


def test_memory_update_appends_without_replacing_existing_notes() -> None:
    existing = ["CT101 runs Home Assistant", "CT102 runs Joplin"]
    response_body = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {"new_notes": ["CT102 runs Joplin and Jellyfin"]}
                    )
                }
            }
        ]
    }
    cfg = SimpleNamespace(
        base_url="http://localhost:8080/v1",
        model="local",
        memory_min_tokens=100,
        max_tokens=300,
        memory_max_tokens=700,
        request_timeout=30,
    )
    response = io.BytesIO(json.dumps(response_body).encode())
    with patch.object(chat_ui.urllib.request, "urlopen", return_value=response):
        updated = chat_ui.update_memory(
            cfg,
            existing,
            "CT102 is running Joplin and Jellyfin",
            "CT102 runs Joplin and Jellyfin",
        )

    assert updated == [
        "CT101 runs Home Assistant",
        "CT102 runs Joplin",
        "CT102 runs Joplin and Jellyfin",
    ]
    assert existing == ["CT101 runs Home Assistant", "CT102 runs Joplin"]


def test_explicit_memory_request_appends_without_changing_old_notes() -> None:
    existing = ["CT101 runs Home Assistant", "CT102 runs Joplin"]

    updated = chat_ui.update_memory(
        Mock(),
        existing,
        "remember that CT103 runs Plex",
        "CT103 runs Plex",
    )

    assert updated == [*existing, "CT103 runs Plex"]
    assert existing == ["CT101 runs Home Assistant", "CT102 runs Joplin"]


def test_tool_call_and_result_share_a_full_width_panel() -> None:
    panel = chat_ui.tool_result_panel(
        "node_status",
        {},
        "free -b; df -hT; uptime; pveversion",
        "safe",
        "host details\n[exit 0]",
        verbose=False,
    )

    assert panel.expand is True
    assert panel.title == "⚡ node_status"
    assert "free -b; df -hT; uptime; pveversion" in panel.renderable.plain
    assert "→ exit 0, 2 lines · /last to expand" in panel.renderable.plain


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
        patch.object(main.subprocess, "Popen", return_value=process) as popen,
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
    process_kwargs = popen.call_args.kwargs
    assert Path(process_kwargs["stdout"].name) == Path("serve.log")
    assert process_kwargs["stderr"] is main.subprocess.STDOUT
    assert process_kwargs["start_new_session"] is True


def test_client_owns_sigint_while_launcher_waits() -> None:
    previous_handler = object()
    observed_handlers = []

    def set_handler(_signum, handler):
        observed_handlers.append(handler)
        return previous_handler

    def run_client(_command, *, cwd):
        assert cwd == Path(".")
        assert observed_handlers == [main.signal.SIG_IGN]
        return 130

    with (
        patch.object(main.signal, "signal", side_effect=set_handler),
        patch.object(main.subprocess, "call", side_effect=run_client),
    ):
        result = main._run_client(["python", "client.py"], Path("."))

    assert result == 130
    assert observed_handlers == [main.signal.SIG_IGN, previous_handler]


def test_chat_picker_redraws_banner_after_search() -> None:
    picker = Mock()
    picker.prompt.side_effect = ["s containers", "q"]
    cfg = Mock()
    with (
        patch.object(chat_ui, "PromptSession", return_value=picker),
        patch.object(chat_ui, "list_saved_chats", return_value=[]),
        patch.object(chat_ui, "print_banner") as print_banner,
        patch.object(chat_ui.console, "print"),
    ):
        result = chat_ui.choose_or_create_chat(cfg, [])

    assert result is None
    assert print_banner.call_count == 2


def test_opening_a_chat_redraws_banner_before_transcript() -> None:
    cfg = Mock()
    selected = (
        "chat-id",
        Path("chat.json"),
        "Example",
        [{"role": "system", "content": "system prompt"}],
    )
    picker = Mock()
    picker.prompt.side_effect = EOFError
    with (
        patch.object(chat_ui, "PromptSession", return_value=picker),
        patch.object(chat_ui, "print_banner") as print_banner,
        patch.object(chat_ui.console, "print"),
    ):
        result = chat_ui.run_chat_session(cfg, ["memory"], selected)

    assert result is False
    print_banner.assert_called_once_with(cfg, 1)
