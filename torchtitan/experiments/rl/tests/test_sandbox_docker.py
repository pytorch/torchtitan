# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import importlib.util

import pytest

from torchtitan.experiments.rl.examples.terminal_bench import (
    TerminalBenchEnv,
    TerminalBenchSample,
    TerminalBenchVerifier,
)
from torchtitan.experiments.rl.sandbox import DockerSandboxClient, SandboxSpec

pytestmark = pytest.mark.docker


def _docker_client_or_skip():
    if importlib.util.find_spec("docker") is None:
        pytest.skip("docker Python package is not installed")
    import docker

    client = docker.from_env()
    try:
        client.ping()
    except Exception as error:
        client.close()
        pytest.skip(f"Docker daemon is unavailable: {error}")
    return client


def test_docker_sandbox_contract(tmp_path) -> None:
    docker_client = _docker_client_or_skip()
    image = "python:3.12-slim"
    docker_client.images.pull(image)
    docker_client.close()

    async def run():
        client = DockerSandboxClient.Config(max_concurrent_sessions=1).build()
        session = await client.create(
            SandboxSpec(
                image=image,
                memory_mb=256,
            ),
            owner_id="docker-contract-test",
        )
        result = await session.exec("printf out; printf err >&2; exit 4")
        dropped_uid = await session.exec(
            "setpriv --reuid nobody --regid nogroup --clear-groups id -u"
        )
        source = tmp_path / "input.txt"
        source.write_text("artifact")
        await session.upload(source, "/tmp/input.txt")
        destination = tmp_path / "downloaded.txt"
        await session.download("/tmp/input.txt", destination)
        sandbox_id = session.id
        await session.close()
        await session.close()
        return result, dropped_uid, destination.read_text(), sandbox_id

    result, dropped_uid, content, sandbox_id = asyncio.run(run())
    assert result.exit_code == 4
    assert result.stdout == "out"
    assert result.stderr == "err"
    assert dropped_uid.exit_code == 0
    assert dropped_uid.stdout.strip() == "65534"
    assert content == "artifact"

    docker_client = _docker_client_or_skip()
    import docker

    with pytest.raises(docker.errors.NotFound):
        docker_client.containers.get(sandbox_id)
    docker_client.close()


def test_terminal_bench_docker_end_to_end(tmp_path) -> None:
    docker_client = _docker_client_or_skip()
    work_context = tmp_path / "work"
    verifier_context = tmp_path / "verifier"
    work_context.mkdir()
    verifier_context.mkdir()
    (work_context / "Dockerfile").write_text(
        "FROM python:3.12-slim\nRUN mkdir -p /app\nWORKDIR /app\n"
    )
    (verifier_context / "Dockerfile").write_text(
        """
FROM python:3.12-slim
RUN mkdir -p /tests /logs/verifier /app
COPY test.sh /tests/test.sh
RUN chmod +x /tests/test.sh
""".strip()
        + "\n"
    )
    (verifier_context / "test.sh").write_text(
        """#!/bin/sh
if [ "$(cat /app/answer.txt 2>/dev/null)" = "42" ]; then
  echo 1 > /logs/verifier/reward.txt
  exit 0
fi
echo 0 > /logs/verifier/reward.txt
exit 1
"""
    )
    work_image = "torchtitan-tb-ci-work:latest"
    verifier_image = "torchtitan-tb-ci-verifier:latest"
    docker_client.images.build(path=str(work_context), tag=work_image, rm=True)
    docker_client.images.build(path=str(verifier_context), tag=verifier_image, rm=True)
    docker_client.close()

    sample = TerminalBenchSample(
        task_name="terminal-bench/ci",
        instruction="Write 42 to /app/answer.txt.",
        artifact_paths=("/app/answer.txt",),
        work_sandbox=SandboxSpec(image=work_image, memory_mb=256),
        verifier_sandbox=SandboxSpec(image=verifier_image, memory_mb=256),
        verifier_timeout_s=30,
    )

    async def run():
        env = TerminalBenchEnv(
            TerminalBenchEnv.Config(),
            env_input=sample,
            sandbox_client=DockerSandboxClient.Config().build(),
        )
        await env.init()
        from renderers.base import ParsedToolCall

        await env.step(
            {
                "role": "assistant",
                "tool_calls": [
                    ParsedToolCall(
                        raw="",
                        name="terminal",
                        arguments={"command": "printf 42 > /app/answer.txt"},
                        id="call-1",
                    )
                ],
            }
        )
        artifacts = await env.export_artifacts(tmp_path / "artifacts")
        await env.close()
        verifier = TerminalBenchVerifier.Config().build(
            sandbox_client=DockerSandboxClient.Config().build()
        )
        return await verifier.verify(
            sample=sample,
            artifacts=artifacts,
            owner_id=env.owner_id,
        )

    try:
        result = asyncio.run(run())
        assert result.reward == 1.0
        assert result.exit_code == 0
    finally:
        docker_client = _docker_client_or_skip()
        for image in (work_image, verifier_image):
            try:
                docker_client.images.remove(image=image, force=True)
            except Exception:
                pass
        docker_client.close()
