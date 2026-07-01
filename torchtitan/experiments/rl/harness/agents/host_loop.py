# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Host-side coding-agent loop: the agent runs on the controller, the sandbox is
only a remote Bash/file backend.

Why this exists (vs ``claude_code.run_claude_code``): the in-sandbox Claude Code
CLI cannot dial back to the inbound-firewalled adapter, so every LLM turn is
relayed host<->sandbox over the Daytona ``fs`` control plane (file-relay bridge):
~0.15s poll-discovery + 3 serial Daytona fs round-trips PER turn, and the whole
~20k-token request is serialized through a Daytona file each turn. With 20-48
turns/rollout the generator sits idle (KV ~5%) waiting on those round-trips.

This module flips it: a small host-side ReAct loop talks
DIRECTLY to the on-box Anthropic adapter over localhost (zero Daytona round-trips
for the LLM call; TITO prompt bridging keeps it one episode), and ships only the
tool commands to the sandbox via ``sb.exec`` / ``read_file`` / ``write_file``.
The big LLM payload stays local; only small tool I/O crosses Daytona.

Token capture, diff capture, grading, binary reward, and zero-std filtering are
all unchanged: the adapter stays the single LLM funnel (keyed by ``Bearer
<session_id>``) and the agent's file writes land in the same sandbox ``workdir``
that ``git_diff`` reads.

Tool legend: Bash -> ``sb.exec``; Read -> ``sb.read_file``; Write ->
``sb.write_file``; Edit -> read-modify-write (Daytona has no edit primitive).
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import time

import aiohttp

from torchtitan.experiments.rl.harness.agents.claude_code import (
    apply_pre_commands,
    ensure_agent_user,
)
from torchtitan.experiments.rl.harness.sandbox import Sandbox

logger = logging.getLogger(__name__)

# Anthropic model name the adapter answers to; arbitrary (the adapter ignores it).
ADAPTER_MODEL_NAME = "titan-actor"

# Per-turn output cap fed to tool_result so a huge grep / test log does not blow
# the model context. The adapter also enforces the model context budget.
_TOOL_OUTPUT_LIMIT = int(os.environ.get("SWE_TOOL_OUTPUT_LIMIT", "16000"))
# Hard ceiling on agent turns (a turn = one LLM call). Mirrors the in-sandbox
# Claude Code budget; the wall-clock ``time_budget_sec`` is the real guard.
_MAX_TURNS = int(os.environ.get("SWE_MAX_TURNS", "60"))

_SYSTEM_PROMPT = os.environ.get(
    "SWE_HOST_SYSTEM_PROMPT",
    "You are an autonomous software engineer fixing a real bug in the repository "
    "at the given working directory. Work in small steps: locate the relevant code "
    "with Bash (grep -rn on class/function names from the issue, or find) -- do NOT "
    "guess file paths. ALWAYS Read a file before you Edit it. Make minimal precise "
    "edits to source files only (do NOT touch tests). If an Edit fails because the "
    "old text does not match, Read the file again and copy the exact text, or "
    "rewrite the whole file with Write. After editing, run the repo's tests with "
    "Bash to verify. When the fix is complete, reply with a short summary and no "
    "further tool calls.",
)

_DEFAULT_TASK_PROMPT = os.environ.get(
    "SWE_HOST_TASK_PROMPT",
    "Resolve the issue described below. Edit source files only; do not modify "
    "tests. Use the project's own interpreter (e.g. `.venv/bin/python -m pytest`) "
    "to run tests, not the system python.\n\n",
)

# Anthropic tool schemas exposed to the policy. Names mirror Claude Code's so a
# Qwen3 policy that has seen Claude-Code-style tools stays in-distribution.
_TOOLS: list[dict] = [
    {
        "name": "Bash",
        "description": (
            "Run a bash command in the repository working directory and return its "
            "stdout/stderr and exit code. Use for searching (grep -rn), listing, and "
            "running tests."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "the bash command"}
            },
            "required": ["command"],
        },
    },
    {
        "name": "Read",
        "description": "Read a file from the sandbox and return its contents with line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "path to the file"}
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "Write",
        "description": "Write (create or overwrite) a file in the sandbox with the given contents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "Edit",
        "description": (
            "Replace the first exact occurrence of old_string with new_string in a "
            "file. old_string must match the file contents exactly (including "
            "indentation). Read the file first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
]


def _truncate(text: str, limit: int = _TOOL_OUTPUT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    return (
        text[:head] + f"\n...[{len(text) - limit} chars truncated]...\n" + text[-tail:]
    )


def _abspath(workdir: str, path: str) -> str:
    return path if path.startswith("/") else f"{workdir.rstrip('/')}/{path}"


async def _tool_bash(sb: Sandbox, workdir: str, inp: dict, timeout: int) -> str:
    cmd = str(inp.get("command", "")).strip()
    if not cmd:
        return "error: empty command"
    # No persistent shell across exec calls -> run each command from workdir.
    wrapped = f"cd {shlex.quote(workdir)} && {cmd}"
    try:
        ec, out, err = await sb.exec(
            wrapped, user="agent", timeout=timeout, check=False
        )
    except Exception as e:  # exec transport failure -> surface to the agent
        return f"error: exec failed: {type(e).__name__}: {e}"
    body = out or ""
    if err:
        body += ("\n" if body else "") + "[stderr]\n" + err
    return _truncate(f"exit_code={ec}\n{body}".rstrip())


async def _tool_read(sb: Sandbox, workdir: str, inp: dict) -> str:
    path = _abspath(workdir, str(inp.get("file_path", "")))
    try:
        content = await sb.read_file(path, user="agent")
    except Exception as e:
        return f"error: could not read {path}: {type(e).__name__}: {e}"
    if not content:
        return f"(file {path} is empty or does not exist)"
    lines = content.splitlines()
    numbered = "\n".join(f"{i + 1}\t{ln}" for i, ln in enumerate(lines))
    return _truncate(numbered)


async def _tool_write(sb: Sandbox, workdir: str, inp: dict) -> str:
    path = _abspath(workdir, str(inp.get("file_path", "")))
    content = inp.get("content", "")
    if not isinstance(content, str):
        content = str(content)
    try:
        await sb.write_file(path, content, user="agent")
    except Exception as e:
        return f"error: could not write {path}: {type(e).__name__}: {e}"
    return f"wrote {len(content)} bytes to {path}"


def _ws_tolerant_span(content: str, old: str) -> tuple[int, int] | None:
    """Locate ``old`` in ``content`` comparing lines after stripping surrounding
    whitespace, so an Edit still applies when the model's ``old_string`` differs
    only in indentation / trailing space (a common failure: the model rewrites the
    snippet from memory with slightly different whitespace). Returns the exact byte
    span of the (unique) match in ``content``, or None when there is no match or
    more than one -- the caller then reports the not-found error rather than editing
    the wrong place. A wrong-but-applied edit is harmless under RL (it just fails
    grading -> reward 0); the win is recovering the whitespace-only mismatches.
    """
    content_lines = content.splitlines(keepends=True)
    old_lines = old.splitlines()
    n = len(old_lines)
    if n == 0:
        return None
    target = [line.strip() for line in old_lines]
    spans: list[tuple[int, int]] = []
    for start in range(len(content_lines) - n + 1):
        window = content_lines[start : start + n]
        if [line.strip() for line in window] == target:
            i = sum(len(line) for line in content_lines[:start])
            spans.append((i, i + sum(len(line) for line in window)))
    return spans[0] if len(spans) == 1 else None


async def _tool_edit(sb: Sandbox, workdir: str, inp: dict) -> str:
    path = _abspath(workdir, str(inp.get("file_path", "")))
    old = inp.get("old_string", "")
    new = inp.get("new_string", "")
    if not isinstance(old, str) or not isinstance(new, str):
        return "error: old_string/new_string must be strings"
    try:
        content = await sb.read_file(path, user="agent")
    except Exception as e:
        return f"error: could not read {path} for edit: {type(e).__name__}: {e}"
    if not old:
        # Empty old_string -> treat as overwrite/append-from-empty (create).
        content = new
    elif old in content:
        count = content.count(old)
        if count > 1:
            return (
                f"error: old_string is not unique in {path} ({count} matches). Add "
                "more surrounding context to make it unique."
            )
        content = content.replace(old, new, 1)
    else:
        # Exact match failed -> whitespace-tolerant fallback (indentation / trailing
        # space only). Recovers the common Edit failure where the model retyped the
        # snippet with slightly different whitespace.
        span = _ws_tolerant_span(content, old)
        if span is None:
            return (
                f"error: old_string not found in {path}. Read the file again and copy "
                "the exact text (including indentation), or use Write to rewrite it."
            )
        i, j = span
        content = content[:i] + new + content[j:]
    try:
        await sb.write_file(path, content, user="agent")
    except Exception as e:
        return f"error: could not write {path}: {type(e).__name__}: {e}"
    return f"edited {path}"


async def _dispatch_tool(
    sb: Sandbox, workdir: str, name: str, inp: dict, *, exec_timeout: int
) -> str:
    if name == "Bash":
        return await _tool_bash(sb, workdir, inp, exec_timeout)
    if name == "Read":
        return await _tool_read(sb, workdir, inp)
    if name == "Write":
        return await _tool_write(sb, workdir, inp)
    if name == "Edit":
        return await _tool_edit(sb, workdir, inp)
    return f"error: unknown tool {name!r}"


async def run_host_loop(
    sb: Sandbox,
    *,
    workdir: str,
    session_id: str,
    adapter_url: str,
    time_budget_sec: int,
    problem_statement: str = "",
    pre_commands: list[str] | str | None = None,
    prompt: str | None = None,
    max_turns: int = _MAX_TURNS,
    exec_timeout: int = 600,
) -> int:
    """Drive a host-side ReAct agent against the on-box adapter; tools run in ``sb``.

    Returns the number of agent turns taken (>=0); negative on setup failure. The
    rollouter ignores the return value and drains captured turns from the adapter
    via ``finish_session`` exactly as for the in-sandbox path.
    """
    await ensure_agent_user(sb, workdir)
    if pre_commands:
        await apply_pre_commands(sb, workdir, pre_commands)
    # Mirror the in-sandbox path: leave the statement on disk too (harmless; the
    # diff excludes it) so a Read of PROBLEM_STATEMENT.md works if the model tries.
    await sb.write_file(
        f"{workdir}/PROBLEM_STATEMENT.md", problem_statement or "", user="agent"
    )

    task_text = (prompt or _DEFAULT_TASK_PROMPT) + (
        f"Working directory: {workdir}\n\n"
        f"--- ISSUE ---\n{problem_statement or ''}\n"
    )
    messages: list[dict] = [{"role": "user", "content": task_text}]

    headers = {
        "Authorization": f"Bearer {session_id}",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    url = adapter_url.rstrip("/") + "/v1/messages"
    deadline = time.time() + time_budget_sec
    turns = 0

    # trust_env=False: the loopback adapter call must ignore the box's fwdproxy.
    async with aiohttp.ClientSession(trust_env=False) as http:
        while turns < max_turns and time.time() < deadline:
            payload = {
                "model": ADAPTER_MODEL_NAME,
                "system": _SYSTEM_PROMPT,
                "messages": messages,
                "tools": _TOOLS,
                "max_tokens": 8192,
                "stream": False,
            }
            # Retry the adapter call: at high rollout concurrency the on-box adapter's
            # accept backlog can transiently drop a connection (empty-error "adapter
            # call failed"); a few reconnects ride it out instead of killing the whole
            # rollout (which would waste its episode + inflate the untrainable rate).
            data = None
            for attempt in range(4):
                try:
                    async with http.post(
                        url,
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(
                            total=max(60, exec_timeout + 300)
                        ),
                    ) as resp:
                        if resp.status != 200:
                            body = await resp.text()
                            logger.warning(
                                "[host_loop] %s: adapter %d: %s",
                                session_id,
                                resp.status,
                                body[:200],
                            )
                            break  # non-200 (e.g. session closed) is not retryable
                        data = await resp.json()
                    break  # success
                except Exception as e:
                    if attempt == 3:
                        logger.warning(
                            "[host_loop] %s: adapter call failed after retries: %r",
                            session_id,
                            e,
                        )
                    else:
                        await asyncio.sleep(0.5 * (attempt + 1))
            if data is None:
                break

            turns += 1
            blocks = data.get("content") or []
            stop_reason = data.get("stop_reason")
            # Echo the assistant turn verbatim so the next request's message prefix
            # hash-matches and the adapter TITO-appends (one packed episode).
            messages.append({"role": "assistant", "content": blocks})

            tool_uses = [
                b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use"
            ]
            if not tool_uses:
                # No tool call -> the agent is done (or a length-capped text turn).
                if stop_reason == "max_tokens":
                    # Nudge it to continue rather than ending on a truncated thought.
                    messages.append(
                        {
                            "role": "user",
                            "content": "(your previous message was truncated; continue)",
                        }
                    )
                    continue
                break

            results: list[dict] = []
            for tu in tool_uses:
                name = tu.get("name", "")
                inp = tu.get("input") or {}
                if not isinstance(inp, dict):
                    inp = {}
                out = await _dispatch_tool(
                    sb, workdir, name, inp, exec_timeout=exec_timeout
                )
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.get("id", ""),
                        "content": out,
                    }
                )
            messages.append({"role": "user", "content": results})

    logger.info("[host_loop] %s: finished after %d turns", session_id, turns)
    return turns
