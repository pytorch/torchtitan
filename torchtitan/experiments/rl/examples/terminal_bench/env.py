# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from uuid import uuid4

from renderers import Message, ToolSpec
from renderers.base import ParsedToolCall

from torchtitan.experiments.rl.environment import (
    MessageEnv,
    MessageEnvInitOutput,
    MessageEnvStepOutput,
)
from torchtitan.experiments.rl.examples.terminal_bench.data import (
    TerminalBenchArtifact,
    TerminalBenchSample,
)
from torchtitan.experiments.rl.sandbox import (
    SandboxClient,
    SandboxExecResult,
    SandboxPathNotFoundError,
    SandboxSession,
)

logger = logging.getLogger(__name__)

TERMINAL_TOOL: ToolSpec = {
    "name": "terminal",
    "description": "Execute a shell command in the task workspace.",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            }
        },
        "required": ["command"],
    },
}

_TASK_PROMPT = (
    "Solve the task in the provided terminal environment. Use the terminal tool "
    "to inspect and modify the workspace. When the task is complete, respond with "
    "a concise final answer.\n\n"
)


class TerminalBenchEnv(MessageEnv):
    """Pattern 2 Terminal-Bench environment with a framework-owned tool loop."""

    @dataclass(kw_only=True, slots=True)
    class Config(MessageEnv.Config):
        max_tool_output_chars: int = 20_000

        def __post_init__(self) -> None:
            if self.max_tool_output_chars < 1:
                raise ValueError(
                    "max_tool_output_chars must be >= 1; "
                    f"got {self.max_tool_output_chars}"
                )

    def __init__(
        self,
        config: Config,
        *,
        env_input: TerminalBenchSample,
        sandbox_client: SandboxClient,
    ) -> None:
        self._config = config
        self._sample = env_input
        self._sandbox_client = sandbox_client
        self._owner_id = f"{env_input.task_name}-{uuid4().hex}"
        self._work_sandbox: SandboxSession | None = None

    @property
    def owner_id(self) -> str:
        return self._owner_id

    async def init(self) -> MessageEnvInitOutput:
        self._work_sandbox = await self._sandbox_client.create(
            self._sample.work_sandbox,
            owner_id=f"{self._owner_id}-work",
        )
        return MessageEnvInitOutput(
            init_prompt_messages=[
                {
                    "role": "user",
                    "content": _TASK_PROMPT + self._sample.instruction,
                }
            ],
            tools=[TERMINAL_TOOL],
        )

    async def step(self, completion_message: Message) -> MessageEnvStepOutput:
        tool_calls: list[ParsedToolCall] = completion_message.get("tool_calls") or []
        if not tool_calls:
            return MessageEnvStepOutput(done=True)
        if self._work_sandbox is None:
            raise RuntimeError("TerminalBenchEnv.init must run before step")

        env_messages: list[Message] = []
        for tool_call in tool_calls:
            result = await self._execute_tool_call(tool_call)
            message: Message = {
                "role": "tool",
                "content": self._format_result(result),
            }
            if tool_call.id:
                message["tool_call_id"] = tool_call.id
            env_messages.append(message)
        return MessageEnvStepOutput(env_messages=env_messages)

    async def export_artifacts(
        self, destination_root: Path
    ) -> tuple[TerminalBenchArtifact, ...]:
        """Copy declared task artifacts out of the live work sandbox."""
        if self._work_sandbox is None:
            raise RuntimeError("TerminalBenchEnv has no work sandbox")

        downloaded: list[TerminalBenchArtifact] = []
        for remote_path in self._sample.artifact_paths:
            local_path = _local_artifact_path(destination_root, remote_path)
            try:
                await self._work_sandbox.download(remote_path, local_path)
            except SandboxPathNotFoundError:
                logger.info(
                    "Terminal-Bench artifact %s was not produced by %s",
                    remote_path,
                    self._sample.task_name,
                )
                continue
            downloaded.append(
                TerminalBenchArtifact(
                    remote_path=remote_path,
                    local_path=local_path,
                )
            )
        return tuple(downloaded)

    async def close(self) -> None:
        if self._work_sandbox is None:
            return
        try:
            await self._work_sandbox.close()
        finally:
            self._work_sandbox = None

    async def _execute_tool_call(self, tool_call: ParsedToolCall) -> SandboxExecResult:
        if tool_call.name != "terminal":
            return SandboxExecResult(
                exit_code=2,
                stderr=f"unknown tool: {tool_call.name}",
            )
        arguments = tool_call.arguments
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = None
        if not isinstance(arguments, dict) or not isinstance(
            arguments.get("command"), str
        ):
            return SandboxExecResult(
                exit_code=2,
                stderr="terminal requires a string 'command' argument",
            )
        assert self._work_sandbox is not None
        return await self._work_sandbox.exec(arguments["command"], cwd="/app")

    def _format_result(self, result: SandboxExecResult) -> str:
        payload = json.dumps(
            {
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
            ensure_ascii=False,
        )
        limit = self._config.max_tool_output_chars
        if len(payload) <= limit:
            return payload
        return "[output truncated]\n" + payload[-limit:]


def _local_artifact_path(root: Path, remote_path: str) -> Path:
    path = PurePosixPath(remote_path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"invalid Terminal-Bench artifact path: {remote_path!r}")
    return root.joinpath(*path.parts[1:])
