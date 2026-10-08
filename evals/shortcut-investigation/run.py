# Copyright (C) 2020-2026, François-Guillaume Fernandez.
# SPDX-License-Identifier: Apache-2.0

"""Run each condition in a fresh Codex CLI session with the same bounded budget."""

import argparse
import json
import os
import signal
import subprocess  # noqa: S404
import sys
import time
from contextlib import suppress
from pathlib import Path


def tool_calls(lines: list[str]) -> int:
    """Count tool-start events from complete JSONL lines.

    Returns:
        Number of command, MCP and web tool starts.
    """
    count = 0
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "item.started" and event.get("item", {}).get("type") in {
            "command_execution",
            "mcp_tool_call",
            "web_search",
        }:
            count += 1
    return count


def run_one(workspace: Path, prompt_path: Path, python: Path) -> dict:
    """Retain traces and distinguish infrastructure/budget failures from completed evaluations.

    Returns:
        Observed status, exit code, tool count and elapsed seconds.
    """
    prompt = (
        f"You are a fresh coding-agent evaluation participant. Work ONLY in {workspace}. "
        f"Use {python} for Python commands. Read and follow {prompt_path} as the common evaluation task, "
        "then complete it. You may read that prompt file only outside your workspace. "
        "Do not access sibling workspaces, graders, or the parent chat. "
        "Maximum four shell/tool invocations and 180 seconds. Write response.json inside your workspace "
        "and return the same JSON. Count your tool invocations."
    )
    command = [
        "codex",
        "exec",
        "--ignore-user-config",
        "--ephemeral",
        "--skip-git-repo-check",
        "--sandbox",
        "workspace-write",
        "-c",
        'approval_policy="never"',
        "--json",
        "-C",
        str(workspace),
        "-",
    ]
    # Uses the installed CLI's default model/auth. Do not inject credentials or change proxy policy.
    started = time.monotonic()
    with (
        (workspace / "trace.jsonl").open("w", encoding="utf-8") as trace,
        (workspace / "stderr.log").open("w", encoding="utf-8") as errors,
    ):
        process = subprocess.Popen(  # noqa: S603
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=errors,
            start_new_session=True,
        )
        process.stdin.write(prompt.encode("utf-8"))
        process.stdin.close()
        # Poll bounded reads, so a silent endpoint failure cannot hang the evaluation.
        import selectors  # noqa: PLC0415

        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        calls, status, pending = 0, "completed", ""
        while True:
            if time.monotonic() - started >= 180:
                status = "timeout"
                break
            if not selector.select(timeout=0.2):
                if process.poll() is not None:
                    break
                continue
            chunk = os.read(process.stdout.fileno(), 65536).decode("utf-8", errors="replace")
            if not chunk:
                break
            trace.write(chunk)
            trace.flush()
            pending += chunk
            lines = pending.split("\n")
            pending = lines.pop()
            calls += tool_calls(lines)
            if calls > 4:
                status = "tool_budget_exceeded"
                break
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        # The leader may have exited while a child still owns the pipe or workspace.
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        selector.close()
        process.stdout.close()
    if status == "completed" and (process.returncode != 0 or not (workspace / "response.json").exists()):
        status = "infrastructure_or_agent_failure"
    record = {
        "status": status,
        "returncode": process.returncode,
        "tool_calls": calls,
        "seconds": time.monotonic() - started,
        "command": command,
    }
    (workspace / "execution.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    arguments = parser.parse_args()
    prompt_file = Path(__file__).with_name("prompt.txt").resolve()
    for case in ("preprocessing", "shortcut", "control"):
        for condition in ("baseline", "extended"):
            target = arguments.root.resolve() / condition / case
            if (target / "response.json").exists():
                raise ValueError("Use fresh prepared workspaces; refusing to overwrite submissions")
            print(case, condition, run_one(target, prompt_file, arguments.python.resolve()), flush=True)  # noqa: T201
