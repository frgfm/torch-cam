#!/usr/bin/env bash
# Copyright (C) 2020-2026, François-Guillaume Fernandez.
# SPDX-License-Identifier: Apache-2.0

# Run fresh Codex sessions with identical tool/time budgets; retain traces and failure records.
set -euo pipefail
[[ $# == 2 ]] || { echo "Usage: $0 ROOT PYTHON" >&2; exit 2; }
for tool in codex jq timeout setsid; do command -v "$tool" >/dev/null; done
root=$(realpath "$1")
python=$(realpath "$2")
prompt=$(realpath "$(dirname "${BASH_SOURCE[0]}")/prompt.txt")
[[ -x "$python" ]]
for workspace in "$root"/{baseline,extended}/{preprocessing,shortcut,control}; do
    [[ -d "$workspace" && ! -e "$workspace/response.json" ]] || {
        echo "Use fresh prepared workspaces: $workspace" >&2; exit 2;
    }
done

pid=""
cleanup() {
    if [[ -n "$pid" ]]; then
        kill -KILL -- "-$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
count_calls() {
    jq -Rn '[inputs | fromjson? | select(.type == "item.started") |
        .item.type | select(. == "command_execution" or . == "mcp_tool_call" or . == "web_search")] | length' "$1"
}

for case in preprocessing shortcut control; do
    for condition in baseline extended; do
        workspace="$root/$condition/$case"
        command=(codex exec --ignore-user-config --ephemeral --skip-git-repo-check
            --sandbox workspace-write -c 'approval_policy="never"' --json -C "$workspace" -)
        started=$SECONDS
        status=completed
        : >"$workspace/trace.jsonl"
        # Foreground timeout keeps descendants in the setsid group for cleanup even after the CLI exits.
        setsid timeout --foreground --kill-after=2s 180s "${command[@]}" \
            >"$workspace/trace.jsonl" 2>"$workspace/stderr.log" <<EOF &
You are a fresh coding-agent evaluation participant. Work ONLY in $workspace.
Use $python for Python commands. Read and follow $prompt as the common task.
You may read that prompt file only outside your workspace. Do not access sibling
workspaces, graders, or the parent chat. Maximum four shell/tool invocations and
180 seconds. Write response.json and return the same JSON. Count tool invocations.
EOF
        pid=$!
        while kill -0 "$pid" 2>/dev/null; do
            if (( $(count_calls "$workspace/trace.jsonl") > 4 )); then
                status=tool_budget_exceeded
                break
            fi
            sleep 0.2
        done
        # Kill surviving children without draining a pipe they might hold open.
        kill -KILL -- "-$pid" 2>/dev/null || true
        returncode=0
        wait "$pid" || returncode=$?
        pid=""
        calls=$(count_calls "$workspace/trace.jsonl")
        if [[ "$status" == completed ]]; then
            if (( calls > 4 )); then status=tool_budget_exceeded
            elif (( returncode == 124 || returncode == 137 )); then status=timeout
            elif (( returncode != 0 )) || [[ ! -f "$workspace/response.json" ]]; then
                status=infrastructure_or_agent_failure
            fi
        fi
        jq -n --arg status "$status" --argjson returncode "$returncode" \
            --argjson tool_calls "$calls" --argjson seconds "$((SECONDS - started))" \
            --argjson command "$(jq -cn --args '$ARGS.positional' -- "${command[@]}")" \
            '{status:$status, returncode:$returncode, tool_calls:$tool_calls, seconds:$seconds, command:$command}' \
            >"$workspace/execution.json"
        printf '%s %s %s\n' "$case" "$condition" "$status"
    done
done
