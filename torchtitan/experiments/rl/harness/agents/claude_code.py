# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Claude Code coding-agent runner over the ``Sandbox`` contract.

This is the pluggable *harness*: an unmodified Claude Code CLI binary runs headless
inside a sandbox and is pointed at our on-box Anthropic adapter (which serves the
trained policy and captures every turn as training tokens). The harness itself is
agent-agnostic plumbing -- swapping Claude Code for another CLI that speaks an
OpenAI/Anthropic-compatible API is a matter of a different run command.

Responsibilities:
  - ``boot_agent_sandbox``: create a fresh sandbox from the task image and install
    the Claude Code CLI by downloading the self-contained binary from its CDN
    inside the sandbox (no-op if the image already ships it).
  - ``ensure_agent_user``: create an unprivileged ``agent`` user that owns workdir.
  - ``run_claude_code``: write the problem statement, (for Daytona) start the
    file-relay bridge, spawn ``claude -p`` pointed at the adapter, poll a done
    marker, and dump the agent trajectory.
  - ``git_diff``: capture the agent's patch for grading.

Env knobs (set by the launcher):
  ``SWE_CLAUDE_CDN``         Claude Code binary CDN base (default Anthropic GCS)
  ``SWE_BOOT_CONCURRENCY``   max simultaneous sandbox boots (default 8)
  ``SWE_BOOT_RETRIES``       retries for a transient boot/install failure (default 2)
  ``SWE_CLAUDE_EXTRA_ARGS``  extra args appended to ``claude -p`` (settings, disallowed tools)
  ``SWE_TRAJECTORY_DUMP_DIR``host dir to copy each claude_code_trajectory.jsonl into
  ``SWE_CC_PROMPT``          the agent's task instruction

Ported from THUDM/slime ``examples/coding_agent_rl/sandbox.py`` (claude runner +
diff capture; R2E grading lives in the example's ``grading.py``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from torchtitan.experiments.rl.harness.sandbox import (
    DaytonaSandbox,
    make_sandbox,
    Sandbox,
)

logger = logging.getLogger(__name__)

SWE_BOOT_CONCURRENCY = int(os.environ.get("SWE_BOOT_CONCURRENCY", "8"))
SWE_BOOT_RETRIES = int(os.environ.get("SWE_BOOT_RETRIES", "2"))
# Claude Code CDN (GCS): the self-contained ``linux-x64/claude`` binary is fetched
# from here INSIDE the sandbox (its own fast egress) instead of uploaded from the
# controller. ``{CDN}/stable`` -> version; ``{CDN}/{version}/linux-x64/claude`` ->
# binary. Mirrors genai/msl/rl claude_code_download.sh.
CLAUDE_CDN = os.environ.get(
    "SWE_CLAUDE_CDN",
    "https://storage.googleapis.com/"
    "claude-code-dist-86c565f3-f756-42ad-8dfa-d59b1c096819/claude-code-releases",
)
CC_PROMPT = os.environ.get(
    "SWE_CC_PROMPT",
    "Read PROBLEM_STATEMENT.md in the current directory and resolve the issue. "
    "Edit source files only (do NOT touch tests). After editing, run the relevant "
    "tests to verify your fix passes. Use the project's OWN interpreter to run "
    "code/tests (e.g. `.venv/bin/python -m pytest ...` from the repo root), NOT "
    "the system `python3` (it lacks the repo's dependencies). Do NOT modify "
    "PROBLEM_STATEMENT.md and do NOT commit. When finished, print a one-line "
    "summary and exit.",
)

# Claude Code's stream-json trajectory: write it under the agent's HOME, OUTSIDE
# the repo workdir, so the agent can't Read its own (huge) trajectory mid-run and
# blow its context. It's untracked either way (git_diff is tracked-only), but the
# read-leak was real (a rollout Read /testbed/claude_code_trajectory.jsonl -> +10k
# tokens -> context overflow).
_TRAJ_PATH = "/home/agent/claude_code_trajectory.jsonl"

# Anthropic model name the adapter answers to; arbitrary (the adapter ignores it).
ADAPTER_MODEL_NAME = "titan-actor"

_BOOT_SEM: asyncio.Semaphore | None = None


@asynccontextmanager
async def boot_agent_sandbox(image: str) -> AsyncIterator[Sandbox]:
    """Boot a fresh sandbox and install the Claude Code toolchain.

    Creates the sandbox from the task image (backend chosen by
    ``TT_SANDBOX_BACKEND``) and installs the Claude Code CLI by downloading the
    self-contained binary from its CDN INSIDE the sandbox (the sandbox's own fast
    egress), retries transient boot failures, and closes the sandbox when the
    caller leaves the context.
    """
    global _BOOT_SEM
    if _BOOT_SEM is None:
        _BOOT_SEM = asyncio.Semaphore(SWE_BOOT_CONCURRENCY)

    sb = None
    last_err: Exception | None = None
    for attempt in range(SWE_BOOT_RETRIES):
        cand = make_sandbox(image)
        try:
            async with _BOOT_SEM:
                await cand.__aenter__()
                try:
                    await install_toolchain(cand)
                except BaseException:
                    await cand.__aexit__(None, None, None)
                    raise
            sb = cand
            break
        except Exception as e:
            last_err = e
            logger.warning(
                "[claude_code] provision attempt %d/%d failed: %s: %s",
                attempt + 1,
                SWE_BOOT_RETRIES,
                type(e).__name__,
                str(e)[:200],
            )
            await asyncio.sleep(1 + attempt)
    if sb is None:
        assert last_err is not None
        raise last_err
    try:
        yield sb
    finally:
        await sb.__aexit__(None, None, None)


async def install_toolchain(sb: Sandbox) -> None:
    """Install the Claude Code CLI by downloading the self-contained binary from
    its CDN INSIDE the sandbox (the sandbox's own fast egress), in ONE exec.

    This is the key latency fix for high-latency controller->sandbox links (e.g.
    cross-region MAST -> daytona-US, ~10x slower than a local box): uploading the
    ~76MB Claude tarball + a Node tarball from the controller and running ~6 install
    execs each cost a slow round-trip and time out ("command execution timeout").
    Instead the sandbox curls the binary from the CDN at datacenter speed (~1s),
    in a single exec. The binary is a self-contained Node SEA, so no separate Node
    install is needed (matches genai/msl/rl claude_code_download.sh). No-op if the
    image already ships a working ``claude``.
    """
    ec, _, _ = await sb.exec("claude --version", user="root", timeout=60, check=False)
    if ec == 0:
        return
    await sb.exec(
        "set -e\n"
        f"ver=$(curl -fsSL {CLAUDE_CDN}/stable)\n"
        f'curl -fsSL -o /usr/local/bin/claude "{CLAUDE_CDN}/$ver/linux-x64/claude"\n'
        "chmod 0755 /usr/local/bin/claude\n"
        "claude --version\n",
        user="root",
        timeout=300,
        check=True,
    )


async def ensure_agent_user(sb: Sandbox, workdir: str) -> None:
    """Create the unprivileged 'agent' user that owns workdir + can git diff.

    A pre-seeded settings file pre-acks bypass-permissions so claude-code starts
    headless without an onboarding prompt.
    """
    await sb.exec(
        f"id agent >/dev/null 2>&1 || useradd -m -s /bin/bash agent && "
        f"chown -R agent:agent /home/agent {workdir} && "
        f"git config --system --add safe.directory '*' && id agent && "
        f"mkdir -p /home/agent/.claude && "
        f'echo \'{{"hasCompletedOnboarding": true, "bypassPermissionsModeAccepted": true}}\' '
        f"| tee /home/agent/.claude.json /home/agent/.claude/settings.json > /dev/null && "
        f"chown -R agent:agent /home/agent/.claude /home/agent/.claude.json",
        user="root",
        check=True,
        timeout=60,
    )
    # R2E images keep the repo interpreter under /root (uv-managed venv symlinked
    # into /root/.local), which the unprivileged 'agent' user cannot traverse or
    # exec -> claude's own `.venv/bin/python` test runs fail. Grant read+exec so
    # the agent can self-verify with the project's python. Best-effort.
    await sb.exec(
        "[ -d /root/.local ] && chmod a+rx /root && chmod -R a+rX /root/.local || true",
        user="root",
        check=False,
        timeout=120,
    )


async def apply_pre_commands(
    sb: Sandbox, workdir: str, pre: list[str] | str, *, user: str = "agent"
) -> None:
    """Run dataset ``pre_commands`` (e.g. ``git checkout <base_sha> -f``).

    Keeps the work sandbox baseline aligned with eval; skipping in the work sandbox
    would make the model's diff context mismatch the eval base -> apply failures.
    """
    body = (
        pre.replace("\\n", "\n")
        if isinstance(pre, str)
        else "\n".join(c for c in (pre or []) if c)
    )
    pre_path = f"{workdir}/__cagent_pre__.sh"
    await sb.write_file(pre_path, "set -e\n" + body, user=user)
    await sb.exec(
        f"chmod 755 {pre_path} && cd {workdir} && bash {pre_path}",
        user=user,
        check=False,
        timeout=600,
    )


async def run_claude_code(
    sb: Sandbox,
    *,
    workdir: str,
    session_id: str,
    adapter_url: str,
    time_budget_sec: int,
    problem_statement: str = "",
    pre_commands: list[str] | str | None = None,
    prompt: str | None = None,
) -> int:
    """Prepare the SWE workspace, write PROBLEM_STATEMENT.md, then run Claude Code.

    For a Daytona sandbox (which cannot dial back to an inbound-firewalled box),
    start the file-relay bridge and point claude at the in-sandbox proxy instead of
    the unreachable host URL. Returns the agent process exit code (``-2`` on budget
    exceeded).
    """
    await ensure_agent_user(sb, workdir)
    if pre_commands:
        await apply_pre_commands(sb, workdir, pre_commands)
    await sb.write_file(
        f"{workdir}/PROBLEM_STATEMENT.md", problem_statement or "", user="agent"
    )

    bridge = None
    cc_adapter_url = adapter_url
    if isinstance(sb, DaytonaSandbox):
        from torchtitan.experiments.rl.harness.sandbox import start_bridge

        bridge = await start_bridge(sb, adapter_url)
        cc_adapter_url = bridge.local_url
    try:
        rc = await _spawn_claude_code(
            sb,
            workdir=workdir,
            session_id=session_id,
            adapter_url=cc_adapter_url,
            prompt=prompt or CC_PROMPT,
            time_budget_sec=time_budget_sec,
        )
        # Pull claude-code's stream-json trajectory out before the sandbox dies so
        # the human-readable agent trace (Read/Edit/Bash/tool turns) survives.
        await dump_trajectory(sb, workdir, session_id)
        return rc
    finally:
        if bridge is not None:
            await bridge.stop()


async def dump_trajectory(sb: Sandbox, workdir: str, session_id: str) -> None:
    """Best-effort copy of ``claude_code_trajectory.jsonl`` from the sandbox to
    ``SWE_TRAJECTORY_DUMP_DIR`` on the host. No-op if the dir is unset."""
    dump_dir = os.environ.get("SWE_TRAJECTORY_DUMP_DIR", "")
    if not dump_dir:
        return
    try:
        traj = await sb.read_file(_TRAJ_PATH, user="agent")
        if not (traj or "").strip():
            return
        os.makedirs(dump_dir, exist_ok=True)
        safe = session_id.replace("/", "_")
        path = os.path.join(dump_dir, f"{safe}.jsonl")
        with open(path, "w") as f:
            f.write(traj)
        logger.info("[claude_code] trajectory dumped: %s (%d bytes)", path, len(traj))
    except Exception as e:
        logger.warning("[claude_code] trajectory dump failed: %s", e)


async def _spawn_claude_code(
    sb: Sandbox,
    *,
    workdir: str,
    session_id: str,
    adapter_url: str,
    prompt: str,
    time_budget_sec: int,
) -> int:
    """Spawn claude-code detached + poll a done-marker file.

    A gateway may reset a long-lived foreground exec, so the launcher writes the
    exit code into a marker file that we poll every 5s via short RPCs (which also
    keeps the sandbox alive against idle GC).
    """
    done = f"{workdir}/.cagent_done"
    launcher = f"{workdir}/.cagent_run.sh"
    traj = _TRAJ_PATH

    launcher_body = (
        "#!/bin/bash\n"
        f"cd {workdir}\n"
        "export HOME=/home/agent\n"
        f"/usr/local/bin/claude -p {json.dumps(prompt)} "
        "--permission-mode bypassPermissions "
        "--output-format stream-json --include-partial-messages "
        "--include-hook-events --verbose "
        f"{os.environ.get('SWE_CLAUDE_EXTRA_ARGS', '').strip()} "
        f"2>&1 | tee {shlex.quote(traj)}\n"
        f"echo $? > {done}\n"
    )
    await sb.write_file(launcher, launcher_body, user="agent")
    await sb.exec(f"chmod +x {launcher}", user="agent", timeout=30)

    env = {
        "ANTHROPIC_BASE_URL": adapter_url,
        "ANTHROPIC_AUTH_TOKEN": session_id,
        "ANTHROPIC_MODEL": ADAPTER_MODEL_NAME,
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
        # The adapter relays the model reply in ONE shot only after the full
        # generation (file-relay bridge cannot stream incrementally). A long
        # generation (big model + up to max_tokens) can exceed the client's
        # default request timeout, making it retry mid-generation -- wasteful and,
        # under high fanout, a source of duplicate in-flight requests. Wait long
        # enough to receive the one-shot reply on the first attempt. Override via
        # SWE_API_TIMEOUT_MS.
        "API_TIMEOUT_MS": os.environ.get("SWE_API_TIMEOUT_MS", "900000"),
    }
    env_keys = ",".join(env.keys())
    await sb.exec(
        f"runuser -u agent --whitelist-environment={env_keys}"
        f" -- bash -c 'setsid {launcher} < /dev/null > /dev/null 2>&1 &'",
        user="root",
        env=env,
        timeout=30,
        check=True,
    )

    deadline = time.time() + time_budget_sec
    exit_code = -2  # convention: -2 = budget exceeded
    while time.time() < deadline:
        await asyncio.sleep(5)
        ec, out, _ = await sb.exec(
            f"test -f {done} && cat {done}", user="agent", timeout=15, check=False
        )
        if ec == 0:
            try:
                exit_code = int((out or "").strip() or "-1")
            except ValueError:
                exit_code = -1
            break
    return exit_code


async def git_diff(
    sb: Sandbox, workdir: str, *, tracked_only: bool = False, user: str = "agent"
) -> str:
    """Capture the agent's patch.

    ``tracked_only=True`` skips ``git add -N .`` so pre-existing untracked
    environment artifacts (e.g. R2E images carry ``datasets/``, ``install.sh``,
    ``run_tests.sh`` in the working tree) are not captured as spurious new-file
    sections that would break ``git apply`` in the evaluator.
    """
    add = "" if tracked_only else "git add -N . && "
    cmd = (
        f"cd {workdir} && {add}"
        f"git diff -- . ':(exclude)PROBLEM_STATEMENT.md' "
        f"':(exclude)claude_code_trajectory.jsonl' "
        f"':(exclude).cagent_done' ':(exclude).cagent_run.sh' "
        f"':(exclude)__cagent_pre__.sh'"
    )
    _, out, _ = await sb.exec(cmd, user=user, timeout=120)
    return out
