#!/usr/bin/env python3
"""Start the local API server, terminal chat, or command-line Proxmox agent."""
from __future__ import annotations

import argparse
import ipaddress
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from pve_config import DEFAULT_CONFIG_PATH, load_settings


console = Console()
ORANGE = "#e56f00"
MUTED = "#808080"


def _choose_mode() -> str:
    console.print(
        Panel(
            "[bold #e56f00]PVE-Chat[/bold #e56f00]\n"
            "[#808080]Choose how you want to run the local model (Ctrl+C to exit)[/#808080]\n\n"
            "[bold]1[/bold]  API server only  [#808080]· connect with any "
            "OpenAI-compatible client[/#808080]\n"
            "[bold]2[/bold]  API server + chat  [#808080]· start the server "
            "and open the terminal chat[/#808080]\n"
            "[bold]3[/bold]  Command-line agent  [#808080]· issue Proxmox "
            "commands without the chat UI[/#808080]",
            title="Proxmox AI",
            border_style=ORANGE,
            padding=(1, 2),
        )
    )
    while True:
        try:
            choice = console.input(Text("❯ ", style=f"bold {ORANGE}") + "Choose 1, 2, or 3 [2]: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[#808080]Cancelled.[/]")
            return "cancel"
        if choice in ("", "2"):
            return "with-chat"
        if choice == "1":
            return "server-only"
        if choice == "3":
            return "agent"
        console.print("[yellow]Please enter 1, 2, or 3.[/]")


def _client_url(host: str, port: int) -> str:
    """Build a loopback URL for a server bound to a local or wildcard address."""
    if host in {"0.0.0.0", "::", ""}:
        host = "127.0.0.1"
    else:
        try:
            address = ipaddress.ip_address(host)
            if address.is_unspecified:
                host = "127.0.0.1"
            elif address.version == 6:
                host = f"[{host}]"
        except ValueError:
            pass
    return f"http://{host}:{port}/v1"


def _wait_until_ready(
    process: subprocess.Popen[bytes],
    url: str,
    timeout: float,
) -> bool:
    deadline = time.monotonic() + timeout
    models_url = f"{url.rstrip('/')}/models"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            request = urllib.request.Request(
                models_url, headers={"Accept": "application/json"}
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(1)
    return False


def _stop_server(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    console.print("\n[#808080]Stopping the local API server…[/]")
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        console.print("[yellow]Server did not stop in time; forcing shutdown.[/]")
        process.kill()
        process.wait()


def _run_server_only(
    script_dir: Path,
    config_path: Path,
    settings: dict,
) -> int:
    server = settings["server"]
    bind_host = server["host"]
    port = server["port"]
    api_url = _client_url(bind_host, port)

    console.print(
        Panel.fit(
            f"[bold #e56f00]Starting OpenAI-compatible API[/]\n"
            f"[#808080]Listening on[/] [bold]{bind_host}:{port}[/]\n"
            f"[#808080]API endpoint[/] [bold]{api_url}[/]\n"
            "[#808080]Press Ctrl+C to stop.[/]",
            border_style=ORANGE,
        )
    )
    if bind_host not in {"127.0.0.1", "localhost", "::1"}:
        console.print(
            "[yellow]The server is bound beyond loopback and does not add "
            "authentication. Only expose it on a trusted network.[/]"
        )

    try:
        return subprocess.call(
            [sys.executable, str(script_dir / "serve.py"), "--config", str(config_path)],
            cwd=script_dir,
        )
    except KeyboardInterrupt:
        console.print("\n[#808080]Server stopped.[/]")
        return 130


def _run_client(client_command: list[str], script_dir: Path) -> int:
    """Let the interactive client handle Ctrl+C without interrupting the launcher."""
    previous_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        return subprocess.call(client_command, cwd=script_dir)
    finally:
        signal.signal(signal.SIGINT, previous_sigint)


def _run_with_client(
    script_dir: Path,
    config_path: Path,
    settings: dict,
    startup_timeout: float,
    client_command: list[str],
    client_name: str,
) -> int:
    server = settings["server"]
    api_url = _client_url(server["host"], server["port"])
    server_command = [
        sys.executable,
        str(script_dir / "serve.py"),
        "--config",
        str(config_path),
    ]

    console.print(
        Panel.fit(
            f"[bold #e56f00]Starting model server and {client_name}[/]\n"
            f"[#808080]API endpoint[/] [bold]{api_url}[/]",
            border_style=ORANGE,
        )
    )
    process: subprocess.Popen[bytes] | None = None
    log_path = script_dir / "serve.log"
    log_file = None
    try:
        log_file = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            server_command,
            cwd=script_dir,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        console.print("[#808080]Loading model; waiting for the API…[/]")
        if not _wait_until_ready(process, api_url, startup_timeout):
            if process.poll() is None:
                console.print(
                    f"[bold red]API did not become ready within "
                    f"{startup_timeout:g} seconds. See {log_path} for server output.[/]"
                )
                return 1
            console.print(
                f"[bold red]Model server exited during startup "
                f"(exit {process.returncode}). See {log_path} for server output.[/]"
            )
            return process.returncode or 1

        console.print(f"[green]API ready; starting {client_name}.[/]")
        return _run_client(client_command, script_dir)
    except KeyboardInterrupt:
        console.print("\n[#808080]Interrupted.[/]")
        return 130
    except OSError as exc:
        console.print(f"[bold red]Could not start a process: {exc}[/]")
        return 1
    finally:
        if process is not None:
            _stop_server(process)
        if log_file is not None:
            log_file.close()


def _run_with_chat(
    script_dir: Path,
    config_path: Path,
    settings: dict,
    startup_timeout: float,
) -> int:
    api_url = _client_url(settings["server"]["host"], settings["server"]["port"])
    client_command = [
        sys.executable,
        str(script_dir / "pve-chat.py"),
        "--config",
        str(config_path),
        "--base-url",
        api_url,
    ]
    return _run_with_client(
        script_dir, config_path, settings, startup_timeout,
        client_command, "chat",
    )


def _run_with_agent(
    script_dir: Path,
    config_path: Path,
    settings: dict,
    startup_timeout: float,
) -> int:
    api_url = _client_url(settings["server"]["host"], settings["server"]["port"])
    client_command = [
        sys.executable,
        str(script_dir / "pve_agent.py"),
        "--config",
        str(config_path),
        "--base-url",
        api_url,
    ]
    return _run_with_client(
        script_dir, config_path, settings, startup_timeout,
        client_command, "command-line agent",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--server-only",
        action="store_true",
        help="run only the OpenAI-compatible API server",
    )
    mode.add_argument(
        "--with-chat",
        action="store_true",
        help="start the API server and then launch the terminal chat",
    )
    mode.add_argument(
        "--agent",
        action="store_true",
        help="start the API server and then launch the command-line agent",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="TOML config file (default: config.toml)",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=300,
        help="seconds to wait for the API with --with-chat or --agent",
    )
    args = parser.parse_args(argv)
    if args.startup_timeout <= 0:
        parser.error("--startup-timeout must be greater than zero")

    config_path = args.config.expanduser().resolve()
    try:
        settings, _ = load_settings(config_path)
    except ValueError as exc:
        parser.error(f"{exc}\nRun `python3 setup_config.py` to create config.toml.")

    script_dir = Path(__file__).resolve().parent
    if args.server_only:
        selected_mode = "server-only"
    elif args.with_chat:
        selected_mode = "with-chat"
    elif args.agent:
        selected_mode = "agent"
    else:
        selected_mode = _choose_mode()

    if selected_mode == "cancel":
        return 130
    if selected_mode == "server-only":
        return _run_server_only(script_dir, config_path, settings)
    if selected_mode == "agent":
        return _run_with_agent(
            script_dir, config_path, settings, args.startup_timeout
        )
    return _run_with_chat(script_dir, config_path, settings, args.startup_timeout)


if __name__ == "__main__":
    raise SystemExit(main())
