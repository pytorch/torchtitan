# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from renderers.base import ParsedToolCall

from torchtitan.experiments.rl.environment import TokenEnv

from torchtitan.experiments.rl.examples.terminal_bench import (
    RewardTerminalBench,
    TerminalBenchDataset,
    TerminalBenchEnv,
    TerminalBenchRollouter,
    TerminalBenchSample,
    TerminalBenchVerifier,
)
from torchtitan.experiments.rl.examples.terminal_bench.config_registry import (
    rl_grpo_qwen3_0_6b_terminal_bench_smoke,
)
from torchtitan.experiments.rl.rubrics import Rubric
from torchtitan.experiments.rl.sandbox import (
    SandboxExecResult,
    SandboxPathNotFoundError,
    SandboxSpec,
)
from torchtitan.experiments.rl.types import Completion


class _FakeSession:
    def __init__(self, role: str, events: list[str]) -> None:
        self.role = role
        self.events = events
        self.id = role
        self.closed = False
        self.uploaded: dict[str, bytes] = {}

    async def exec(self, command, *, cwd=None, timeout_s=None):
        self.events.append(f"{self.role}:exec:{command}")
        if self.role == "work":
            return SandboxExecResult(exit_code=3, stdout="out", stderr="err")
        if command.startswith("cat "):
            return SandboxExecResult(exit_code=0, stdout="1\n")
        return SandboxExecResult(exit_code=1, stderr="tests failed as expected")

    async def download(self, remote_path: str, local_path: Path) -> None:
        self.events.append(f"work:download:{remote_path}")
        if remote_path == "/logs/artifacts":
            raise SandboxPathNotFoundError(remote_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(b"answer")

    async def upload(self, local_path: Path, remote_path: str) -> None:
        self.events.append(f"verifier:upload:{remote_path}")
        self.uploaded[remote_path] = local_path.read_bytes()

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.events.append(f"{self.role}:close")


class _FakeSandboxClient:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.sessions: list[_FakeSession] = []

    async def create(self, spec, *, owner_id):
        role = "verifier" if owner_id.endswith("-verifier") else "work"
        self.events.append(f"create:{role}:{spec.image}")
        session = _FakeSession(role, self.events)
        self.sessions.append(session)
        return session


def _sample() -> TerminalBenchSample:
    return TerminalBenchSample(
        task_name="terminal-bench/example",
        instruction="Write /app/answer.txt.",
        artifact_paths=("/app/answer.txt", "/logs/artifacts"),
        work_sandbox=SandboxSpec(image="work"),
        verifier_sandbox=SandboxSpec(image="verifier"),
        verifier_timeout_s=30,
    )


def _terminal_call(command: str) -> ParsedToolCall:
    return ParsedToolCall(
        raw="",
        name="terminal",
        arguments={"command": command},
        id="call-1",
    )


def test_terminal_env_exports_artifacts_for_isolated_verifier(tmp_path) -> None:
    client = _FakeSandboxClient()
    env = TerminalBenchEnv(
        TerminalBenchEnv.Config(), env_input=_sample(), sandbox_client=client
    )
    verifier = TerminalBenchVerifier.Config().build(sandbox_client=client)

    async def run():
        init = await env.init()
        step = await env.step(
            {"role": "assistant", "tool_calls": [_terminal_call("do work")]}
        )
        done = await env.step({"role": "assistant", "content": "done"})
        artifacts = await env.export_artifacts(tmp_path)
        await env.close()
        result = await verifier.verify(
            sample=_sample(),
            artifacts=artifacts,
            owner_id=env.owner_id,
        )
        return init, step, done, result

    init, step, done, result = asyncio.run(run())
    assert init.tools[0]["name"] == "terminal"
    tool_result = json.loads(step.env_messages[0]["content"])
    assert tool_result == {"exit_code": 3, "stdout": "out", "stderr": "err"}
    assert step.env_messages[0]["tool_call_id"] == "call-1"
    assert done.done is True
    assert result.as_reward_signals() == {
        "terminal_bench": 1.0,
        "verifier_exit_code": 1.0,
    }
    assert client.sessions[1].uploaded == {"/app/answer.txt": b"answer"}
    assert client.events.index("work:close") < client.events.index(
        "create:verifier:verifier"
    )
    assert all(session.closed for session in client.sessions)


def test_terminal_env_reports_invalid_tool_arguments() -> None:
    client = _FakeSandboxClient()
    env = TerminalBenchEnv(
        TerminalBenchEnv.Config(), env_input=_sample(), sandbox_client=client
    )

    async def run():
        await env.init()
        call = ParsedToolCall(raw="", name="terminal", arguments="not-json", id="bad")
        output = await env.step({"role": "assistant", "tool_calls": [call]})
        await env.close()
        return output

    output = asyncio.run(run())
    result = json.loads(output.env_messages[0]["content"])
    assert result["exit_code"] == 2
    assert "command" in result["stderr"]


def test_terminal_bench_dataset_loads_task_metadata(tmp_path) -> None:
    task = tmp_path / "example"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "environment" / "Dockerfile").write_text("FROM scratch\n")
    (task / "tests" / "Dockerfile").write_text("FROM scratch\n")
    (task / "instruction.md").write_text("solve it")
    (task / "task.toml").write_text(
        """
artifacts = ["/app/answer.txt"]
[task]
name = "terminal-bench/example"
[agent]
timeout_sec = 20
[verifier]
environment_mode = "separate"
timeout_sec = 10
[environment]
cpus = 2
memory_mb = 3072
storage_mb = 4096
network_mode = "public"
""".strip()
    )
    dataset = TerminalBenchDataset.Config(
        tasks_dir=str(tmp_path),
        task_names=["example"],
        image_prefix="registry/tb",
        shuffle=False,
    ).build()
    sample = next(dataset)
    assert sample.task_name == "terminal-bench/example"
    assert sample.work_sandbox.image == "registry/tb-example-work:latest"
    assert sample.verifier_sandbox.image == "registry/tb-example-verifier:latest"
    assert sample.work_sandbox.num_cpus == 2
    assert sample.artifact_paths == ("/app/answer.txt", "/logs/artifacts")


def test_terminal_bench_rollouter_uses_verifier_rubric() -> None:
    config = TerminalBenchRollouter.Config()
    assert isinstance(config.rubric.reward_fns[0], RewardTerminalBench.Config)


def test_terminal_bench_smoke_config_runs_one_two_gpu_step() -> None:
    config = rl_grpo_qwen3_0_6b_terminal_bench_smoke()
    assert isinstance(config.rollouter, TerminalBenchRollouter.Config)
    assert config.async_loop.num_training_steps == 1
    assert config.async_loop.num_prompts_per_train_step == 1
    assert config.async_loop.num_samples_per_prompt == 1
    assert config.trainer.parallelism.tensor_parallel_degree == 1
    assert config.generator.parallelism.tensor_parallel_degree == 1


class _FakeRenderer:
    def render_ids(self, **kwargs):
        return [1]

    def parse_response(self, **kwargs):
        return SimpleNamespace(
            content="done",
            reasoning_content=None,
            tool_calls=None,
        )


async def _generate(prompt_token_ids, **kwargs):
    return Completion(
        min_policy_version=1,
        max_policy_version=1,
        request_id=kwargs["request_id"],
        token_ids=[2],
        token_logprobs=[-0.1],
        finish_reason="stop",
    )


def test_terminal_bench_rollouter_closes_work_before_verification() -> None:
    client = _FakeSandboxClient()
    rollouter = object.__new__(TerminalBenchRollouter)
    rollouter._message_env_config = TerminalBenchEnv.Config()
    rollouter._token_env_config = TokenEnv.Config()
    rollouter._sandbox_client = client
    rollouter._verifier = TerminalBenchVerifier.Config().build(sandbox_client=client)
    rollouter.rubric = Rubric.Config(
        reward_fns=[RewardTerminalBench.Config()],
        error_reward=0.0,
    ).build()
    rollouter.advantage_estimator = lambda group: [0.0] * len(group.rollouts)

    group = asyncio.run(
        rollouter.run_group_rollouts(
            generate_fn=_generate,
            sample=_sample(),
            group_id=2,
            group_size=1,
            sampling=SimpleNamespace(seed=None),
            renderer=_FakeRenderer(),
        )
    )

    assert group.rollouts[0].reward == 1.0
    assert group.rollouts[0].turns[-1].env_rewards == {
        "terminal_bench": 1.0,
        "verifier_exit_code": 1.0,
    }
    assert client.events.index("work:close") < client.events.index(
        "create:verifier:verifier"
    )
