from __future__ import annotations

import importlib.util
import io
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import hardware_tuning
import main
import serve
import setup_config
from hardware_tuning import choose_recommendation, performance_candidates
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
    assert settings["server"]["chat_format"] == "auto"


def test_server_uses_native_chat_template_and_disables_thinking(
    tmp_path: Path,
) -> None:
    settings, _ = load_settings(EXAMPLE_CONFIG_PATH, apply_env_overrides=False)
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"model")
    settings["server"]["model_path"] = str(model_path)
    settings["server"]["chat_format"] = "auto"
    config_path = tmp_path / "config.toml"

    with (
        patch.object(serve, "load_settings", return_value=(settings, config_path)),
        patch.object(serve, "Settings") as server_settings,
        patch.object(serve, "create_app"),
        patch.object(serve.uvicorn, "run"),
    ):
        assert serve.main([]) == 0

    assert server_settings.call_args.kwargs["chat_format"] is None
    assert server_settings.call_args.kwargs["chat_template_kwargs"] == {
        "enable_thinking": False
    }


def test_windows_memory_detection_uses_global_memory_status() -> None:
    def set_memory_status(status_pointer) -> bool:
        status = status_pointer._obj
        status.ullAvailPhys = 768 * hardware_tuning.MIB
        status.ullTotalPhys = 4096 * hardware_tuning.MIB
        return True

    kernel32 = SimpleNamespace(
        GlobalMemoryStatusEx=Mock(side_effect=set_memory_status)
    )
    with patch.object(
        hardware_tuning.ctypes,
        "WinDLL",
        create=True,
        return_value=kernel32,
    ):
        assert hardware_tuning._detect_windows_memory() == (768, 4096)


def test_detect_memory_uses_windows_api_when_proc_meminfo_is_unavailable() -> None:
    with (
        patch.object(hardware_tuning.os, "name", "nt"),
        patch.object(hardware_tuning.Path, "read_text", side_effect=OSError),
        patch.object(
            hardware_tuning,
            "_detect_windows_memory",
            return_value=(768, 4096),
        ) as detect_windows_memory,
    ):
        assert hardware_tuning.detect_memory() == (768, 4096)

    detect_windows_memory.assert_called_once_with()


def test_linux_amd_gpu_detection_reads_drm_memory(tmp_path: Path) -> None:
    device = tmp_path / "card0" / "device"
    device.mkdir(parents=True)
    (device / "vendor").write_text("0x1002", encoding="ascii")
    (device / "device").write_text("0x67df", encoding="ascii")
    (device / "mem_info_vram_total").write_text(
        str(8 * hardware_tuning.MIB * 1024), encoding="ascii"
    )
    (device / "mem_info_vram_used").write_text(
        str(2 * hardware_tuning.MIB * 1024), encoding="ascii"
    )
    (device / "product_name").write_text("Radeon RX 580", encoding="utf-8")

    assert hardware_tuning._detect_linux_amd_gpu(tmp_path) == {
        "name": "Radeon RX 580",
        "total_mib": 8192,
        "free_mib": 6144,
    }


def test_linux_amd_gpu_detection_ignores_non_amd_adapters(tmp_path: Path) -> None:
    device = tmp_path / "card0" / "device"
    device.mkdir(parents=True)
    (device / "vendor").write_text("0x10de", encoding="ascii")

    assert hardware_tuning._detect_linux_amd_gpu(tmp_path) is None


def test_windows_dxgi_uses_correct_adapter_method_slots() -> None:
    assert hardware_tuning.DXGI_ADAPTER1_GET_DESC1_INDEX == 10
    assert hardware_tuning.DXGI_ADAPTER3_QUERY_VIDEO_MEMORY_INFO_INDEX == 14


def test_performance_candidates_compare_gpu_layers_and_cpu_threads() -> None:
    candidates = performance_candidates(
        layers=40,
        max_cpu_threads=8,
        batch_threads=6,
        resource_mode="gpu+cpu",
        gpu_offload_supported=True,
    )

    assert {candidate["n_gpu_layers"] for candidate in candidates} == {0, 20, -1}
    assert {candidate["n_threads"] for candidate in candidates} == {1, 2, 4, 8}
    assert all(
        candidate["n_threads_batch"] == 6
        for candidate in candidates
    )


def test_performance_candidates_respect_selected_resource_mode() -> None:
    cpu_candidates = performance_candidates(
        layers=40,
        max_cpu_threads=4,
        batch_threads=4,
        resource_mode="cpu-only",
        gpu_offload_supported=True,
    )
    gpu_candidates = performance_candidates(
        layers=40,
        max_cpu_threads=4,
        batch_threads=4,
        resource_mode="gpu-only",
        gpu_offload_supported=True,
    )

    assert {candidate["n_gpu_layers"] for candidate in cpu_candidates} == {0}
    assert {candidate["n_gpu_layers"] for candidate in gpu_candidates} == {-1}


def test_speed_search_chooses_fastest_without_reducing_quality_settings(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"model")
    settings = {
        "server": {
            "model_path": str(model_path),
            "n_ctx": 8192,
            "kv_type": "q4_0",
            "n_threads_batch": 6,
        }
    }
    recommendation = {
        "n_ctx": 131072,
        "n_gpu_layers": -1,
        "n_threads_batch": 8,
    }
    metadata = {
        "layers": 42,
        "kv_heads": 2,
        "key_length": 128,
        "value_length": 128,
        "context_length": 131072,
    }

    def fake_benchmark(
        _settings,
        _config_path,
        candidate,
        threads,
        *,
        report,
    ):
        assert report
        assert _settings["server"]["kv_type"] == "q4_0"
        assert _config_path.name == "config.toml"
        assert candidate["n_ctx"] == 8192
        gpu_layers = 42 if candidate["n_gpu_layers"] < 0 else candidate["n_gpu_layers"]
        return {
            count: gpu_layers * 10 + count for count in threads
        }

    with (
        patch(
            "llama_cpp.llama_cpp.llama_supports_gpu_offload",
            return_value=True,
        ),
        patch.object(
            hardware_tuning,
            "benchmark_thread_counts",
            side_effect=fake_benchmark,
        ),
    ):
        fastest = hardware_tuning.find_fastest_configuration(
            settings,
            tmp_path / "config.toml",
            recommendation,
            metadata=metadata,
            gpu={"name": "Test GPU", "total_mib": 8192, "free_mib": 7000},
            model_size_mib=1000,
            ram_budget_mib=4096,
            resource_mode="gpu+cpu",
            max_cpu_threads=4,
            report=Mock(),
        )

    assert fastest == {
        "n_gpu_layers": -1,
        "n_threads": 4,
        "n_threads_batch": 6,
        "n_ctx": 8192,
    }
    assert settings["server"]["n_ctx"] == 8192
    assert settings["server"]["kv_type"] == "q4_0"


def test_speed_benchmark_uses_native_template_without_json_grammar(
    tmp_path: Path,
) -> None:
    class FakeModel:
        def __init__(self) -> None:
            self._ctx = SimpleNamespace(set_n_threads=Mock())
            self.chat_format = "minicpm"
            self.chat_handler = self.native_handler
            self.native_template_kwargs = []
            self.chat_calls = []

        def native_handler(self, **kwargs):
            self.native_template_kwargs.append(kwargs.pop("enable_thinking", None))
            return self.generate(**kwargs)

        def create_chat_completion(self, **kwargs):
            self.chat_calls.append(kwargs)
            return self.chat_handler(**kwargs)

        @staticmethod
        def generate(**kwargs):
            if not kwargs.get("stream"):
                return {}
            return iter(
                [
                    {
                        "choices": [{
                            "delta": {"content": "x"},
                            "finish_reason": None,
                        }]
                    }
                    for _ in range(20)
                ]
            )

        def close(self) -> None:
            pass

    fake_model = FakeModel()
    ticks = iter(index / 20 for index in range(60))
    with (
        patch("llama_cpp.Llama", return_value=fake_model) as llama,
        patch.object(hardware_tuning.time, "perf_counter", side_effect=lambda: next(ticks)),
    ):
        rates = hardware_tuning.benchmark_thread_counts(
            {
                "server": {
                    "model_path": "model.gguf",
                    "kv_type": "q4_0",
                    "n_batch": 128,
                    "n_ubatch": 32,
                    "chat_format": "auto",
                }
            },
            tmp_path / "config.toml",
            {
                "n_gpu_layers": -1,
                "n_ctx": 8192,
                "n_threads": 4,
                "n_threads_batch": 4,
            },
            [4],
            report=Mock(),
        )

    assert rates == {4: 20.0}
    assert len(fake_model.chat_calls) == 4
    assert all("enable_thinking" not in call for call in fake_model.chat_calls)
    assert fake_model.native_template_kwargs == [False] * 4
    assert all(call["messages"][0]["role"] == "user" for call in fake_model.chat_calls)
    assert llama.call_args.kwargs["chat_format"] is None


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


def test_chat_message_panel_shows_tokens_per_second_in_subtitle() -> None:
    with patch.object(chat_ui, "Markdown", side_effect=lambda content: content):
        panel = chat_ui.chat_message_panel(
            "Agent", "Answer", "agent_fg", "agent_bg", 19.36
        )

    assert panel.subtitle.plain == "~19.4 tok/s"


def test_stream_chat_emits_generation_speed_for_sse_chunks() -> None:
    class FakeResponse:
        headers = {"Content-Type": "text/event-stream"}

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def __iter__(self):
            chunks = [
                {"choices": [{"delta": {"content": "hello"}}]},
                {"choices": [{"delta": {"content": " world"}}]},
            ]
            for chunk in chunks:
                yield f"data: {json.dumps(chunk)}\n\n".encode()
            yield b"data: [DONE]\n\n"

    cfg = SimpleNamespace(
        model="local",
        temperature=0,
        max_tokens=100,
        base_url="http://localhost:8080/v1",
        request_timeout=30,
    )
    with (
        patch.object(chat_ui.urllib.request, "urlopen", return_value=FakeResponse()) as urlopen,
        patch.object(chat_ui.time, "perf_counter", side_effect=[1.0, 2.0]),
    ):
        events = list(chat_ui.stream_chat(cfg, []))

    assert events == [
        ("token", "hello"),
        ("token", " world"),
        ("speed", 1.0),
        ("done", ""),
    ]
    payload = json.loads(urlopen.call_args.args[0].data)
    assert "response_format" not in payload
    assert "chat_template_kwargs" not in payload


def test_stream_turn_displays_unwrapped_nested_answer() -> None:
    answer = "Ready. Ask me anything - install packages, manage containers, check status."
    reply = json.dumps({"answer": json.dumps({"answer": answer})})

    class FakeStatus:
        def start(self):
            pass

        def stop(self):
            pass

    class FakeLive:
        def __init__(self, *_args, **_kwargs):
            self.updates = []

        def start(self):
            pass

        def update(self, renderable, **_kwargs):
            self.updates.append(renderable)

        def stop(self):
            pass

    fake_live = FakeLive()
    with (
        patch.object(chat_ui.console, "status", return_value=FakeStatus()),
        patch.object(chat_ui, "Live", return_value=fake_live),
        patch.object(
            chat_ui,
            "chat_message_panel",
            side_effect=lambda _name, content, *_args, **_kwargs: content,
        ),
        patch.object(
            chat_ui,
            "stream_chat",
            return_value=iter([("token", reply), ("done", "")]),
        ),
    ):
        raw_reply, _speed = chat_ui.stream_turn(SimpleNamespace(), [])

    assert raw_reply == reply
    assert fake_live.updates[-1] == answer


def test_stream_turn_recovers_answer_from_invalid_json_envelope() -> None:
    answer = "Full host report:\n\n- Memory: 8.15GB total"
    reply = (
        '{"answer": "Full host report:\n\n- Memory: 8.15GB total", '
        '"tool": "node_status", "arguments": {}}'
    )

    class FakeStatus:
        def start(self):
            pass

        def stop(self):
            pass

    class FakeLive:
        def __init__(self, *_args, **_kwargs):
            self.updates = []

        def start(self):
            pass

        def update(self, renderable, **_kwargs):
            self.updates.append(renderable)

        def stop(self):
            pass

    fake_live = FakeLive()
    with (
        patch.object(chat_ui.console, "status", return_value=FakeStatus()),
        patch.object(chat_ui, "Live", return_value=fake_live),
        patch.object(
            chat_ui,
            "chat_message_panel",
            side_effect=lambda _name, content, *_args, **_kwargs: content,
        ),
        patch.object(
            chat_ui,
            "stream_chat",
            return_value=iter([("token", reply), ("done", "")]),
        ),
    ):
        normalized_reply, _speed = chat_ui.stream_turn(SimpleNamespace(), [])

    assert json.loads(normalized_reply) == {"answer": answer}
    assert fake_live.updates[-1] == answer


def test_restored_transcript_displays_saved_tokens_per_second() -> None:
    message = {
        "role": "assistant",
        "content": '{"answer":"Hello"}',
        "tokens_per_second": 19.36,
    }
    with patch.object(chat_ui, "print_chat_message") as print_chat_message:
        chat_ui.show_saved_transcript([message])

    print_chat_message.assert_called_once_with(
        "◆ Agent", "Hello", "agent_fg", "agent_bg", 19.36
    )


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
        tokens_per_second=None,
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
        tokens_per_second=None,
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


def test_tool_result_panel_shows_tokens_per_second_subtitle() -> None:
    panel = chat_ui.tool_result_panel(
        "node_status",
        {},
        "uptime",
        "safe",
        "host details\n[exit 0]",
        tokens_per_second=19.36,
    )

    assert panel.subtitle.plain == "~19.4 tok/s"


def test_restored_tool_result_displays_saved_tokens_per_second() -> None:
    result = "container list\n[exit 0]"
    messages = [
        {
            "role": "assistant",
            "content": '{"tool":"list_guests","arguments":{}}',
            "tokens_per_second": 8.25,
        },
        {"role": "user", "content": f"[tool list_guests result]\n{result}"},
    ]
    with (
        patch.object(chat_ui, "show_tool_result") as show_result,
        patch.object(chat_ui.console, "print"),
    ):
        chat_ui.show_saved_transcript(messages)

    assert show_result.call_args.kwargs["tokens_per_second"] == 8.25


def test_run_turn_attaches_generation_speed_to_tool_command() -> None:
    cfg = SimpleNamespace(max_steps=3, dry_run=True)
    messages = [{"role": "system", "content": "system"}]
    tool_reply = '{"tool":"list_guests","arguments":{}}'
    final_reply = '{"answer":"Here are the guests."}'
    with (
        patch.object(chat_ui, "compact_messages"),
        patch.object(
            chat_ui,
            "stream_turn",
            side_effect=[(tool_reply, 20.0), (final_reply, 8.0)],
        ),
        patch.object(chat_ui, "show_tool_result") as show_result,
    ):
        answer = chat_ui.run_turn(cfg, messages)

    assert answer == "Here are the guests."
    show_result.assert_called_once_with(
        "list_guests",
        {},
        "pvesh get /cluster/resources --type vm --output-format json",
        "safe",
        "[dry-run] would run: pvesh get /cluster/resources --type vm --output-format json",
        tokens_per_second=20.0,
    )
    assert messages[1]["tokens_per_second"] == 20.0


def test_run_turn_retries_capability_refusal_for_server_status_request() -> None:
    cfg = SimpleNamespace(max_steps=4, dry_run=True)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "send me a full server status report"},
    ]
    replies = [
        ('{"answer":"My capabilities are limited to specific tool calls."}', 19.0),
        ('{"tool":"node_status","arguments":{}}', 20.0),
        ('{"answer":"Here is the server report."}', 18.0),
    ]
    with (
        patch.object(chat_ui, "compact_messages"),
        patch.object(chat_ui, "stream_turn", side_effect=replies),
        patch.object(chat_ui, "show_tool_result") as show_result,
    ):
        answer = chat_ui.run_turn(cfg, messages)

    assert answer == "Here is the server report."
    assert any(
        message.get("role") == "user"
        and "That refusal is incorrect" in message.get("content", "")
        for message in messages
    )
    assert show_result.call_args.args[0] == "node_status"
    assert show_result.call_args.kwargs["tokens_per_second"] == 20.0


def test_run_turn_accepts_plain_text_answer_without_retry() -> None:
    cfg = SimpleNamespace(max_steps=3, dry_run=True)
    messages = [{"role": "system", "content": "system"}]
    reply = "I will provide the report now."
    with (
        patch.object(chat_ui, "compact_messages"),
        patch.object(chat_ui, "stream_turn", return_value=(reply, 18.6)) as stream_turn,
        patch.object(chat_ui, "print_chat_message") as print_message,
    ):
        assert chat_ui.run_turn(cfg, messages) == reply

    stream_turn.assert_called_once()
    print_message.assert_called_once_with(
        "◆ Agent", reply, "agent_fg", "agent_bg", 18.6
    )
    assert json.loads(messages[-1]["content"]) == {"answer": reply}
    assert messages[-1]["tokens_per_second"] == 18.6


def test_run_turn_does_not_execute_tool_when_reply_also_has_answer() -> None:
    cfg = SimpleNamespace(max_steps=8, dry_run=False)
    messages = [{"role": "system", "content": "system"}]
    answer = "Full host status report: memory and disk details."
    reply = json.dumps({
        "answer": answer,
        "tool": "shell",
        "arguments": {"command": "echo unexpected command"},
    })
    with (
        patch.object(chat_ui, "compact_messages"),
        patch.object(chat_ui, "stream_turn", return_value=(reply, 19.0)) as stream_turn,
        patch.object(chat_ui, "show_tool_result") as show_result,
        patch.object(chat_ui, "print_chat_message") as print_message,
    ):
        assert chat_ui.run_turn(cfg, messages) == answer

    stream_turn.assert_called_once()
    show_result.assert_not_called()
    print_message.assert_not_called()
    assert json.loads(messages[-1]["content"]) == {"answer": answer}


def test_run_turn_unwraps_nested_json_answer() -> None:
    cfg = SimpleNamespace(max_steps=3, dry_run=True)
    messages = [{"role": "system", "content": "system"}]
    answer = "Ready. Ask me anything - install packages, manage containers, check status."
    reply = json.dumps({"answer": json.dumps({"answer": answer})})
    with (
        patch.object(chat_ui, "compact_messages"),
        patch.object(chat_ui, "stream_turn", return_value=(reply, 18.0)),
    ):
        assert chat_ui.run_turn(cfg, messages) == answer

    assert json.loads(messages[-1]["content"]) == {"answer": answer}


def test_run_turn_accepts_plain_text_after_tool_without_looping() -> None:
    cfg = SimpleNamespace(max_steps=8, dry_run=True)
    messages = [{"role": "system", "content": "system"}]
    tool_reply = '{"tool":"list_guests","arguments":{}}'
    final_reply = "Here is the full report, with all requested details."
    with (
        patch.object(chat_ui, "compact_messages"),
        patch.object(
            chat_ui,
            "stream_turn",
            side_effect=[
                (tool_reply, 20.0),
                (final_reply, 18.2),
            ],
        ) as stream_turn,
        patch.object(chat_ui, "show_tool_result"),
        patch.object(chat_ui, "print_chat_message") as print_message,
    ):
        answer = chat_ui.run_turn(cfg, messages)

    assert answer == final_reply
    assert stream_turn.call_count == 2
    print_message.assert_called_once_with(
        "◆ Agent", final_reply, "agent_fg", "agent_bg", 18.2
    )
    final_message = messages[-1]
    assert json.loads(final_message["content"]) == {"answer": final_reply}
    assert final_message["tokens_per_second"] == 18.2


def test_run_turn_stops_after_repeated_tool_call_retry() -> None:
    cfg = SimpleNamespace(max_steps=8, dry_run=True)
    messages = [{"role": "system", "content": "system"}]
    repeated_tool_call = '{"tool":"list_guests","arguments":{}}'
    with (
        patch.object(chat_ui, "compact_messages"),
        patch.object(
            chat_ui,
            "stream_turn",
            return_value=(repeated_tool_call, 18.0),
        ) as stream_turn,
        patch.object(chat_ui, "show_tool_result"),
        patch.object(chat_ui, "print_chat_message") as print_message,
    ):
        answer = chat_ui.run_turn(cfg, messages)

    assert answer == (
        "I couldn't complete the request because the model repeated the same "
        "tool call instead of using its result."
    )
    assert stream_turn.call_count == 3
    print_message.assert_called_once()
    assert json.loads(messages[-1]["content"]) == {"answer": answer}


def test_restored_transcript_hides_invalid_reply_followed_by_json_retry() -> None:
    messages = [
        {"role": "user", "content": "send me a full server status report"},
        {
            "role": "assistant",
            "content": "Here is the complete report.",
            "tokens_per_second": 18.6,
        },
        {"role": "user", "content": "Reply with exactly one JSON object."},
        {
            "role": "assistant",
            "content": '{"answer":"Here is the complete report."}',
            "tokens_per_second": 18.2,
        },
    ]
    with (
        patch.object(chat_ui, "print_chat_message") as print_message,
        patch.object(chat_ui.console, "print"),
    ):
        chat_ui.show_saved_transcript(messages)

    print_message.assert_any_call(
        "◆ Agent", "Here is the complete report.", "agent_fg", "agent_bg", 18.2
    )
    assert print_message.call_count == 2
    assert all(
        call.args[-1] != 18.6
        for call in print_message.call_args_list
    )


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
