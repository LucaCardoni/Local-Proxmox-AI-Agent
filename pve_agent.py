#!/usr/bin/env python3
"""pve-agent — natural-language Proxmox VE management via a local LLM.

The model is not trusted to run commands. It only *proposes* a tool call as
JSON; this program validates it, shows it to you (unless --yes is given for
safe commands), runs it over SSH, and feeds the result back.

Talk to it like a person: "how much RAM is the zfs pool using", "start the
jellyfin container", "exec 'docker ps' inside CT 105", "show me /etc/pve".
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass

from pve_config import load_settings


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass
class Config:
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    request_timeout: int
    ssh_host: str
    ssh_user: str
    ssh_timeout: int
    ssh_connect_timeout: int
    ssh_batch_mode: bool
    ssh_strict_host_key_checking: str
    max_steps: int
    context_detection_timeout: int
    memory_min_tokens: int
    memory_max_tokens: int
    confirm: bool = True
    dry_run: bool = False
    verbose: bool = False


# --------------------------------------------------------------------------
# SSH
# --------------------------------------------------------------------------
def ssh_run(cfg: Config, command: str) -> tuple[int, str, str]:
    """Run *command* on the Proxmox host over SSH. BatchMode: key auth only."""
    if cfg.dry_run:
        return 0, f"[dry-run] would run: {command}\n", ""
    argv = [
        "ssh",
        "-o", f"BatchMode={'yes' if cfg.ssh_batch_mode else 'no'}",
        "-o", f"ConnectTimeout={cfg.ssh_connect_timeout}",
        "-o", f"StrictHostKeyChecking={cfg.ssh_strict_host_key_checking}",
        f"{cfg.ssh_user}@{cfg.ssh_host}",
        f"LC_ALL=C.UTF-8 {command}",        # ← kills perl locale warnings
    ]
    try:
        p = subprocess.run(argv, capture_output=True, text=True,
                           timeout=cfg.ssh_timeout)
    except subprocess.TimeoutExpired:
        return 124, "", f"ssh timed out after {cfg.ssh_timeout}s"
    return p.returncode, p.stdout, p.stderr


# --------------------------------------------------------------------------
# LLM client (OpenAI-compatible, llama-server / llama_cpp.server)
# --------------------------------------------------------------------------
def llm_chat(cfg: Config, messages: list[dict]) -> str:
    payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "stop": ["</s>", "<|im_end|>"],
        "response_format": {"type": "json_object"},          # ← NEW: force valid JSON
        "chat_template_kwargs": {"enable_thinking": False},  # ← NEW: disable <think>
    }
    req = urllib.request.Request(
        f"{cfg.base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=cfg.request_timeout) as r:
            body = json.loads(r.read())
    except urllib.error.URLError as e:
        raise SystemExit(f"cannot reach llama-server at {cfg.base_url}: {e}")
    return body["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------
# Safety classification
# --------------------------------------------------------------------------
BLOCKED = [
    (r"\brm\s+(-[a-zA-Z]+\s+)*-[a-zA-Z]*r[a-zA-Z]*f?\s+/(\s|$)", "rm -rf /"),
    (r"\bdd\b[^\n]*\bof=/dev/[sh]d[a-z]", "raw dd to disk"),
    (r"\bmkfs(\.[a-z0-9]+)?\b", "mkfs"),
    (r":\(\)\s*\{.*\};\s*:", "fork bomb"),
    (r"\bpvecm\s+delnode\b", "removing a cluster node"),
    (r"\bwipefs\b", "wipefs"),
]
DANGEROUS = [
    r"\brm\b", r"\brmdir\b", r"\bshred\b",
    r"\bqm\s+destroy\b", r"\bpct\s+destroy\b", r"\bpveam\s+remove\b",
    r"\bzfs\s+destroy\b", r"\bzpool\s+destroy\b",
    # package management: install and upgrade can break a working system
    r"\bapt(-get)?\s+(install|reinstall|remove|purge|autoremove|dist-upgrade|upgrade|full-upgrade)\b",
    r"\bdpkg\s+(-i|--install|-r|--remove|-P|--purge)\b",
    r"\bsystemctl\s+(stop|disable|mask|restart)\b",
    r"\bshutdown\b", r"\breboot\b", r"\bhalt\b",
]


def classify(command: str) -> str:
    for pat, label in BLOCKED:
        if re.search(pat, command):
            return f"blocked:{label}"
    for pat in DANGEROUS:
        if re.search(pat, command):
            return "dangerous"
    return "safe"


def confirm(cfg: Config, desc: str, command: str, level: str) -> bool:
    if level.startswith("blocked:"):
        print(f"\n✗ REFUSED ({level[8:]}): {command}")
        return False
    if cfg.dry_run:
        print(f"\n[dry-run] {desc}\n  $ {command}")
        return False
    if level == "safe" and not cfg.confirm:
        return True
    if level == "safe":
        print(f"\n  → {desc}\n    $ {command}")
        return True
    print(f"\n⚠ DANGEROUS: {desc}\n    $ {command}")
    try:
        return input("    Type 'yes' to run: ").strip().lower() == "yes"
    except EOFError:
        return False


# --------------------------------------------------------------------------
# Helpers to turn Proxmox's fixed-width tables into JSON
# --------------------------------------------------------------------------
def _table_to_rows(text: str) -> list[dict]:
    """Split a fixed-width PVE table into dicts by column positions.

    Position-based slicing (not whitespace-splitting) because PVE tables
    often have empty cells — e.g. the Lock column in `pct list` — which
    break whitespace-based parsing. Column boundaries are taken from the
    header; each row is sliced from one header start to the next.
    """
    lines = [l for l in text.strip().splitlines() if l.strip()]
    if len(lines) < 2:
        return []

    header = lines[0]
    col_starts: list[tuple[int, str]] = []   # (start_pos, column_name)
    i = 0
    while i < len(header):
        while i < len(header) and header[i] == " ":
            i += 1
        if i >= len(header):
            break
        start = i
        while i < len(header):
            if header[i] == " " and i + 1 < len(header) and header[i + 1] == " ":
                break
            i += 1
        name = header[start:i].strip()
        if name:
            col_starts.append((start, name))

    if not col_starts:
        return []

    rows = []
    for line in lines[1:]:
        row = {}
        for idx, (start, name) in enumerate(col_starts):
            end = col_starts[idx + 1][0] if idx + 1 < len(col_starts) else len(line)
            value = line[start:end].strip() if start < len(line) else ""
            row[name] = value
        rows.append(row)
    return rows

def _format_result(rc: int, out: str, err: str) -> str:
    body = out.strip() or "(no stdout)"
    if err.strip():
        clean_err = "\n".join(
            l for l in err.strip().splitlines()
            if not l.startswith("perl: warning:")
            and "are supported and installed on your system" not in l
            and "Falling back to a fallback locale" not in l
            and "Please check that your locale" not in l
        ).strip()
        if clean_err:
            body += f"\n[stderr]\n{clean_err}"
    return body + f"\n[exit {rc}]"


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
def tool_list_guests(cfg: Config, _args: dict) -> str:
    if cfg.dry_run:
        return _run_and_format(
            cfg, "List VMs and containers",
            "pvesh get /cluster/resources --type vm --output-format json")

    rc, out, err = ssh_run(
        cfg, "pvesh get /cluster/resources --type vm --output-format json")
    if rc == 0 and out.strip().startswith(("[", "{")):
        try:
            data = json.loads(out)
        except json.JSONDecodeError:
            return _format_result(rc, out, err)
        # Trim to the fields the model actually needs
        slim = [
            {"vmid": r.get("vmid"), "type": r.get("type"),
             "name": r.get("name"), "status": r.get("status"),
             "maxmem": r.get("maxmem"), "maxdisk": r.get("maxdisk")}
            for r in data if isinstance(r, dict)
        ]
        return json.dumps(slim, indent=2) + f"\n[exit {rc}]"
    # fallback: text mode
    # fallback: text mode
    result = {"vms": [], "cts": []}
    for label, cmd in (("vms", "qm list"), ("cts", "pct list")):
        rc, out, err = ssh_run(cfg, cmd)
        rows = _table_to_rows(out)
        result[label] = rows
        if not rows and (out or err).strip():
            result[f"{label}_raw"] = (out or err).strip()[:500]
    return json.dumps(result, indent=2)


def tool_guest_status(cfg: Config, args: dict) -> str:
    vmid = int(args["vmid"])
    kind = args.get("type", "lxc")
    cli = "qm" if kind == "qemu" else "pct"
    return _run_and_format(cfg, f"{cli} status {vmid}", f"{cli} status {vmid}")


def tool_guest_config(cfg: Config, args: dict) -> str:
    vmid = int(args["vmid"])
    kind = args.get("type", "lxc")
    cli = "qm" if kind == "qemu" else "pct"
    return _run_and_format(cfg, f"{cli} config {vmid}", f"{cli} config {vmid}")


def tool_node_status(cfg: Config, _args: dict) -> str:
    if cfg.dry_run:
        return _run_and_format(
            cfg, "Node status", "free -b; df -hT; uptime; pveversion")

    result = {}

    # Memory: `free -b` line 1 is
    #   "Mem: <total> <used> <free> <shared> <buff/cache> <available>"
    rc, out, _ = ssh_run(cfg, "free -b")
    lines = [l for l in out.strip().splitlines() if l.strip()]
    if len(lines) >= 2:
        parts = lines[1].split()
        if len(parts) >= 7:
            try:
                total, _u, _f, _s, _bc, avail = (int(x) for x in parts[1:7])
                result["memory"] = {
                    "total_gb":     round(total / 1e9, 2),
                    "used_gb":      round((total - avail) / 1e9, 2),
                    "available_gb": round(avail / 1e9, 2),
                    "used_pct":     round(100.0 * (total - avail) / total, 2),
                }
            except ValueError as e:
                result["memory_error"] = f"parse failed: {e}"

    # Disk: df -hT parsed to dicts
    rc, out, _ = ssh_run(cfg, "df -hT -x tmpfs -x devtmpfs")
    rows = _table_to_rows(out)
    if rows:
        result["disk"] = [
            {
                "mount":      r.get("Mounted on") or r.get("Mounted"),
                "filesystem": r.get("Filesystem"),
                "type":       r.get("Type"),
                "size":       r.get("Size"),
                "used":       r.get("Used"),
                "avail":      r.get("Avail"),
                "used_pct":   r.get("Use%"),
            } for r in rows
        ]

    rc, out, _ = ssh_run(cfg, "uptime -p")
    result["uptime"] = out.strip().removeprefix("up ")

    rc, out, _ = ssh_run(cfg, "pveversion")
    result["pveversion"] = out.strip()

    return json.dumps(result, indent=2) + "\n[exit 0]"

def _fmt_gb(n: float) -> float:
    """Bytes → GB (decimal, matches what users expect from disk tools)."""
    return round(n / 1_000_000_000, 2)


def tool_storage_list(cfg: Config, _args: dict) -> str:
    if cfg.dry_run:
        return _run_and_format(
            cfg, "Storage pools",
            "pvesh get /cluster/resources --type storage --output-format json")

    rc, out, err = ssh_run(
        cfg, "pvesh get /cluster/resources --type storage --output-format json")
    if rc != 0 or not out.strip().startswith(("[", "{")):
        rc, out, err = ssh_run(cfg, "pvesm status")
        rows = _table_to_rows(out)
        if not rows:
            return _format_result(rc, out, err)
        return json.dumps({"storage": rows, "exit": rc}, indent=2)

    try:
        raw = json.loads(out)
    except json.JSONDecodeError:
        return _format_result(rc, out, err)

    slim = []
    for s in raw:
        if not isinstance(s, dict):
            continue
        used = s.get("disk", 0) or 0
        total = s.get("maxdisk", 0) or 0
        pct = s.get("disk-usage")
        if pct is None and total > 0:
            pct = round(100.0 * used / total, 2)
        slim.append({
            "storage":     s.get("storage"),
            "type":        s.get("plugintype") or s.get("type"),
            "status":      "active" if s.get("status") == "available" else s.get("status"),
            "used_gb":     _fmt_gb(used),
            "total_gb":    _fmt_gb(total),
            "used_pct":    pct,
        })
    return json.dumps(slim, indent=2) + f"\n[exit {rc}]"


def tool_guest_start(cfg: Config, args: dict) -> str:
    vmid = int(args["vmid"])
    kind = args.get("type", "lxc")
    cli = "qm" if kind == "qemu" else "pct"
    return _run_and_format(cfg, f"Start {kind} {vmid}", f"{cli} start {vmid}")


def tool_guest_stop(cfg: Config, args: dict) -> str:
    vmid = int(args["vmid"])
    kind = args.get("type", "lxc")
    cli = "qm" if kind == "qemu" else "pct"
    return _run_and_format(cfg, f"Stop {kind} {vmid}",
                           f"{cli} shutdown {vmid}")


def tool_guest_exec(cfg: Config, args: dict) -> str:
    vmid = int(args["vmid"])
    inner = args["command"]
    cmd = f"pct exec {vmid} -- bash -lc {shlex.quote(inner)}"
    return _run_and_format(cfg, f"Exec in CT {vmid}: {inner}", cmd)


def tool_shell(cfg: Config, args: dict) -> str:
    cmd = args["command"]
    return _run_and_format(cfg, f"Shell: {cmd}", cmd)


def tool_read_file(cfg: Config, args: dict) -> str:
    path = args["path"]
    cmd = (f"test -f {shlex.quote(path)} && cat {shlex.quote(path)} || "
           f"echo 'not a file: {path}'")
    return _run_and_format(cfg, f"Read {path}", cmd, level="safe")


def _run_and_format(cfg: Config, desc: str, command: str,
                    level: str | None = None) -> str:
    level = level or classify(command)
    if not confirm(cfg, desc, command, level):
        return "REFUSED by safety layer."
    rc, out, err = ssh_run(cfg, command)
    return _format_result(rc, out, err)


TOOL_IMPLS = {
    "list_guests":    tool_list_guests,
    "guest_status":   tool_guest_status,
    "guest_config":   tool_guest_config,
    "node_status":    tool_node_status,
    "storage_list":   tool_storage_list,
    "guest_start":    tool_guest_start,
    "guest_stop":     tool_guest_stop,
    "guest_exec":     tool_guest_exec,
    "shell":          tool_shell,
    "read_file":      tool_read_file,
}

TOOL_SCHEMA = textwrap.dedent("""\
    - list_guests        {}                                   → all VMs + CTs (JSON)
    - guest_status       {"vmid": int, "type": "qemu"|"lxc"}  → running state
    - guest_config       {"vmid": int, "type": "qemu"|"lxc"}  → full config
    - node_status        {}                                   → RAM, disk, uptime (JSON)
    - storage_list       {}                                   → storage pools + usage (JSON)
    - guest_start        {"vmid": int, "type": "qemu"|"lxc"}
    - guest_stop         {"vmid": int, "type": "qemu"|"lxc"}  → clean shutdown
    - guest_exec         {"vmid": int, "command": str}        → run inside LXC
    - shell              {"command": str}                     → raw on Proxmox host
    - read_file          {"path": str}                        → cat a file
""")


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------
def system_prompt(cfg: Config) -> str:
    return f"""You are pve-agent, a general assistant for a Proxmox VE host.
You talk to the user naturally, and you act by calling tools that run
commands over SSH. You can install packages, manage services, inspect and
control VMs and containers, read logs, and answer general questions.

Tools (call ONE per turn):
{TOOL_SCHEMA}
Reply with EXACTLY ONE JSON object, nothing else:
  Tool call:  {{"tool": "<name>", "arguments": {{...}}}}
  Finished:   {{"answer": "<short reply>"}}

For general tasks (install, services, logs, diagnostics), use `shell`.
`shell` runs on the Proxmox host as {cfg.ssh_user} — be specific and cautious.
For tasks INSIDE a container, use `guest_exec`.
For specific queries, use the matching tool directly (for example
`node_status`, `storage_list`, or `list_guests`). NEVER put a tool name
inside a `shell` command; tool names are pve-agent actions, not host commands.
For a host report, call `node_status` as its own tool call. To add other
details, make separate `shell` calls for actual host commands.

Multi-step tasks work: each tool result comes back to you and you can call
another tool. Keep each step small and verify before declaring success.

EXAMPLES:

User: install nginx
You: {{"tool": "shell", "arguments": {{"command": "apt-get update && apt-get install -y nginx"}}}}
[tool result]
You: {{"tool": "shell", "arguments": {{"command": "systemctl is-active nginx"}}}}
[tool result]
You: {{"answer": "nginx installed and running."}}

User: restart container 102
You: {{"tool": "guest_stop", "arguments": {{"vmid": 102, "type": "lxc"}}}}
[tool result]
You: {{"tool": "guest_start", "arguments": {{"vmid": 102, "type": "lxc"}}}}
[tool result]
You: {{"answer": "Container 102 restarted."}}

User: how much disk are my storage pools using
You: {{"tool": "storage_list", "arguments": {{}}}}

User: are there failed services
You: {{"tool": "shell", "arguments": {{"command": "systemctl --failed"}}}}

User: what kernel am I running
You: {{"tool": "shell", "arguments": {{"command": "uname -r"}}}}

User: how are you?
You: {{"answer": "Ready. Ask me anything — install packages, manage containers, check status."}}

User: how much RAM am I using
You: {{"tool": "node_status", "arguments": {{}}}}

User: how full is /
You: {{"tool": "shell", "arguments": {{"command": "df -h /"}}}}

User: show me disk usage on the host
You: {{"tool": "node_status", "arguments": {{}}}}

END EXAMPLES.

Rules:
- If the user is just chatting or asking something that needs no host
  state, answer directly with {{"answer": ...}} — do not call a tool.
- Never invent VMIDs; call list_guests if unsure.
- Never invoke a listed tool by putting its name in a shell command.
- Never call the same tool with the same arguments twice.
- If a command was refused, report it; do not retry.
- When a tool returns used_gb / total_gb / used_pct, quote those numbers
  verbatim. Do NOT convert units or do arithmetic.
- If the user asks for information about the host (RAM, disk, uptime,
  kernel, services, files, processes), USE A TOOL. Never say "I can't
  check X" — the `shell` tool can run ANY command on the host.
- Keep answers to 1-3 sentences.

SSH target: {cfg.ssh_user}@{cfg.ssh_host}
"""


# --------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------
def parse_action(text: str) -> dict | None:
    """Pull the first balanced JSON object out of the model's reply."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text,
                  flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = None
    depth = 0
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    start = None
    return None


# --------------------------------------------------------------------------
# Agent loop
# --------------------------------------------------------------------------
def run_agent(cfg: Config, user_input: str) -> None:
    messages = [
        {"role": "system", "content": system_prompt(cfg)},
        {"role": "user", "content": user_input},
    ]
    calls: list[tuple[str, str]] = []      # (tool, canonical args) history

    for step in range(1, cfg.max_steps + 1):
        # Force an answer near the end so we never spin forever.
        if step >= cfg.max_steps - 1:
            messages.append({"role": "user", "content":
                'Stop calling tools. Reply now with '
                '{"answer": "<best answer from what you already know>"}.'})

        if cfg.verbose:
            print(f"\n[step {step}/{cfg.max_steps}] asking model...")
        reply = llm_chat(cfg, messages)
        if cfg.verbose:
            print(f"[raw] {reply!r}")

        action = parse_action(reply)
        if action is None:
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content":
                "Your reply was not valid JSON. Respond with exactly one "
                "JSON object as instructed."})
            continue

        if "answer" in action:
            print(f"\n{action['answer']}")
            return

        tool = action.get("tool")
        args = action.get("arguments") or {}
        if tool not in TOOL_IMPLS:
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content":
                f"Unknown tool {tool!r}. Use one of: "
                f"{', '.join(TOOL_IMPLS)}."})
            continue

        # Loop detection: identical tool + identical args → nudge hard.
        key = (tool, json.dumps(args, sort_keys=True, default=str))
        if key in calls:
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content":
                f"You already called {tool} with those exact arguments. "
                f"The result above IS the answer. Do not call it again. "
                f'Respond now with {{"answer": "..."}}.'})
            calls.append(key)
            continue

        try:
            result = TOOL_IMPLS[tool](cfg, args)
        except (KeyError, ValueError, TypeError) as e:
            result = f"ERROR: bad arguments for {tool}: {e}"
        calls.append(key)

        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content":
            f"[tool {tool} result]\n{result}"})

    print(f"\n[gave up after {cfg.max_steps} steps]")


def repl(cfg: Config) -> None:
    print("pve-agent — type your request, or 'quit'.")
    print(f"  model:  {cfg.base_url} ({cfg.model})")
    print(f"  target: ssh {cfg.ssh_user}@{cfg.ssh_host}")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line or line.lower() in ("quit", "exit", "q"):
            return
        run_agent(cfg, line)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("prompt", nargs="*", help="one-shot prompt; omit for REPL")
    p.add_argument("--config", help="path to TOML config (default: config.toml)")
    p.add_argument("--base-url")
    p.add_argument("--model")
    p.add_argument("--ssh-host")
    p.add_argument("--ssh-user")
    p.add_argument("--ssh-timeout", type=int)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--temperature", type=float)
    p.add_argument("--max-tokens", type=int)
    p.add_argument("--yes", action="store_true",
                   help="auto-confirm safe tools (dangerous tools still prompt)")
    p.add_argument("--dry-run", action="store_true",
                   help="show every command, run nothing")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)

    try:
        settings, _ = load_settings(a.config)
    except ValueError as exc:
        p.error(str(exc))
    llm = settings["llm"]
    ssh = settings["ssh"]
    agent = settings["agent"]
    cfg = Config(
        base_url=a.base_url or llm["base_url"],
        model=a.model or llm["model"],
        temperature=(a.temperature if a.temperature is not None
                     else llm["temperature"]),
        max_tokens=(a.max_tokens if a.max_tokens is not None
                    else agent["max_tokens"]),
        request_timeout=llm["request_timeout"],
        ssh_host=a.ssh_host or ssh["host"],
        ssh_user=a.ssh_user or ssh["user"],
        ssh_timeout=(a.ssh_timeout if a.ssh_timeout is not None else ssh["timeout"]),
        ssh_connect_timeout=ssh["connect_timeout"],
        ssh_batch_mode=ssh["batch_mode"],
        ssh_strict_host_key_checking=ssh["strict_host_key_checking"],
        max_steps=(a.max_steps if a.max_steps is not None else agent["max_steps"]),
        context_detection_timeout=settings["chat"]["context_detection_timeout"],
        memory_min_tokens=settings["chat"]["memory_min_tokens"],
        memory_max_tokens=settings["chat"]["memory_max_tokens"],
        confirm=not (a.yes or agent["auto_confirm_safe"]),
        dry_run=a.dry_run, verbose=a.verbose,
    )

    if a.prompt:
        run_agent(cfg, " ".join(a.prompt))
        return 0
    repl(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())