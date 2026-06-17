# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SweEnv: the multi-turn coding-agent loop, in message space.

The env IS the agent loop. Each assistant turn calls the ``bash`` tool (run a
command in the sandbox) or ``submit`` (declare done). The env runs the command
and returns the combined output as a tool observation; the model is the agent.
The rollout ends when the agent submits, stops calling tools, or hits the turn
cap -- and the terminal step grades the work IN the still-live sandbox (the
sandbox is gone by scoring time, so grading must happen here). The reward flows
out via ``env_rewards`` and is read back by ``RewardR2EGym``.

This env is task-agnostic about the agent loop; grading is delegated to
``grading.grade_r2e`` (R2E-Gym). A different grader is the seam for SWE-bench.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from renderers import Message, ToolSpec
from renderers.base import ParsedToolCall, ToolCallParseStatus

from torchtitan.experiments.rl.environment import (
    MessageEnv,
    MessageEnvInitOutput,
    MessageEnvStepOutput,
)
from torchtitan.experiments.rl.examples.swe import grading
from torchtitan.experiments.rl.examples.swe.data import R2EGymSample
from torchtitan.experiments.rl.examples.swe.sandbox import Sandbox, SandboxFactory

logger = logging.getLogger(__name__)


BASH_TOOL: ToolSpec = {
    "name": "bash",
    "description": (
        "Run a bash command in the repository sandbox and get its combined "
        "stdout+stderr and exit code. Each call is independent and starts at the "
        "repository root -- shell state (cwd, variables) does NOT persist between "
        "calls, so use absolute paths or chain commands with '&&'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "cmd": {
                "type": "string",
                "description": "The bash command to run, e.g. \"sed -n '1,40p' src/foo.py\".",
            },
        },
        "required": ["cmd"],
    },
}

SUBMIT_TOOL: ToolSpec = {
    "name": "submit",
    "description": (
        "Call this with no arguments when you have finished editing the source "
        "files and believe the issue is resolved. This ends the session and runs "
        "the hidden tests."
    ),
    "parameters": {"type": "object", "properties": {}},
}


def _truncate(text: str, max_chars: int) -> str:
    """Keep a bash observation under ``max_chars``, preserving the tail (where the
    exit status, tracebacks, and test summaries live) over the head."""
    if len(text) <= max_chars:
        return text
    head = max_chars // 4
    tail = max_chars - head
    return f"{text[:head]}\n... [{len(text) - max_chars} chars truncated] ...\n{text[-tail:]}"


def _command_from_tool_call(tool_call: ParsedToolCall) -> str | None:
    """Pull the ``cmd`` argument out of a renderer-parsed ``bash`` tool call."""
    arguments = tool_call.arguments
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if isinstance(arguments, dict):
        cmd = arguments.get("cmd")
        return cmd if isinstance(cmd, str) else None
    return None


class SweEnv(MessageEnv):
    """Multi-turn coding-agent env over a ``Sandbox``, graded by R2E-Gym tests."""

    @dataclass(kw_only=True, slots=True)
    class Config(MessageEnv.Config):
        sandbox: SandboxFactory.Config = field(default_factory=SandboxFactory.Config)
        """Backend selection for the work/eval sandbox."""

        max_turns: int = 30
        """Self-imposed cap on assistant turns. On reaching it the env grades and
        ends, so the rollout still gets a reward (unlike a TokenEnv truncation,
        which would skip grading). Keep <= TokenEnv.max_num_turns if both are set."""

        bash_timeout_s: float = 120.0
        """Per-``bash``-call wall-clock timeout."""

        eval_timeout_s: float = 600.0
        """Wall-clock timeout for the terminal hidden-test run. Must be comfortably
        below TokenEnv.step_timeout_s, which wraps the whole grading step."""

        max_obs_chars: int = 4000
        """Per-call cap on the bash observation returned to the model."""

        def __post_init__(self) -> None:
            if self.max_turns < 1:
                raise ValueError("SweEnv.Config.max_turns must be >= 1")

    def __init__(
        self,
        config: Config,
        *,
        env_input: R2EGymSample,
        sandbox_factory: SandboxFactory | None = None,
    ) -> None:
        self._config = config
        self._sample = env_input
        # The factory (and its concurrency semaphore) is a RUNTIME object, injected
        # by the rollouter so it is shared across the whole run. Fall back to a
        # per-env factory for standalone use (tests, single-env scripts).
        self._factory = sandbox_factory or config.sandbox.build()
        self._repo_root = self._factory.repo_root
        self._sandbox: Sandbox | None = None
        self._turn = 0
        self._graded: dict[str, float] | None = None

    async def init(self) -> MessageEnvInitOutput:
        self._sandbox = await self._factory.provision(image=self._sample.image)
        prompt = (
            "You are an autonomous software engineer fixing a bug in a repository.\n\n"
            f"<issue>\n{self._sample.problem_statement}\n</issue>\n\n"
            f"The repository is checked out at {self._repo_root}. Use the `bash` "
            "tool to explore and edit the source files. When you are confident the "
            "issue is fixed, call the `submit` tool.\n\n"
            "Notes:\n"
            "- Network access is disabled; all dependencies are already installed.\n"
            "- Do NOT edit test files; the hidden tests are run by the harness.\n"
            "- Each bash call starts fresh at the repository root."
        )
        return MessageEnvInitOutput(
            init_prompt_messages=[{"role": "user", "content": prompt}],
            tools=[BASH_TOOL, SUBMIT_TOOL],
        )

    async def step(self, completion_message: Message) -> MessageEnvStepOutput:
        self._turn += 1
        tool_calls: list[ParsedToolCall] = completion_message.get("tool_calls") or []
        submitted = any(tc.name == "submit" for tc in tool_calls)

        # Terminal: the agent submitted, gave a final answer with no tool call, or
        # ran out of turn budget. Grade the live sandbox before it is torn down.
        if submitted or not tool_calls or self._turn >= self._config.max_turns:
            return MessageEnvStepOutput(done=True, env_rewards=await self._grade())

        env_messages: list[Message] = []
        for tc in tool_calls:
            env_messages.append(await self._handle_tool_call(tc))
        return MessageEnvStepOutput(env_messages=env_messages, done=False)

    async def _handle_tool_call(self, tool_call: ParsedToolCall) -> Message:
        if tool_call.status != ToolCallParseStatus.OK:
            return {"role": "tool", "content": f"malformed tool call: {tool_call.raw}"}
        if tool_call.name != "bash":
            return {"role": "tool", "content": f"unknown tool: {tool_call.name}"}
        cmd = _command_from_tool_call(tool_call)
        if cmd is None:
            return {
                "role": "tool",
                "content": "bash tool call missing string 'cmd' argument",
            }
        assert self._sandbox is not None  # provisioned in init()
        res = await self._sandbox.exec(cmd, timeout_s=self._config.bash_timeout_s)
        status = "timed out" if res.timed_out else f"exit={res.exit_code}"
        # An empty body reads identically to a successful command with output the
        # model didn't expect; label it so the agent can tell "ran, found nothing"
        # apart from "produced output" and adapt instead of repeating the command.
        body = _truncate(res.output, self._config.max_obs_chars).rstrip()
        if not body:
            body = "(no output)"
        return {"role": "tool", "content": f"({status})\n{body}"}

    async def _grade(self) -> dict[str, float]:
        """Run the hidden tests once, in the live sandbox. Idempotent."""
        if self._graded is not None:
            return self._graded
        assert self._sandbox is not None  # provisioned in init()
        self._graded = await grading.grade_r2e(
            self._sandbox,
            sample=self._sample,
            repo_root=self._repo_root,
            timeout_s=self._config.eval_timeout_s,
        )
        return self._graded

    async def close(self) -> None:
        # Idempotent; the rollouter closes every env in a finally.
        if self._sandbox is not None:
            await self._sandbox.close()
            self._sandbox = None
