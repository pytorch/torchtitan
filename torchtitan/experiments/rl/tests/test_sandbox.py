# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from torchtitan.experiments.rl.sandbox import (
    daytona as daytona_backend,
    DaytonaSandboxClient,
    SandboxSpec,
)


def test_sandbox_spec_validates_resources() -> None:
    with pytest.raises(ValueError, match="image"):
        SandboxSpec(image="")
    with pytest.raises(ValueError, match="num_cpus"):
        SandboxSpec(image="image", num_cpus=0)
    with pytest.raises(ValueError, match="memory_mb"):
        SandboxSpec(image="image", memory_mb=0)
    with pytest.raises(ValueError, match="storage_mb"):
        SandboxSpec(image="image", storage_mb=0)
    with pytest.raises(ValueError, match="timeout_s"):
        SandboxSpec(image="image", timeout_s=0)


def test_daytona_adapter_maps_spec_and_cleans_up(monkeypatch) -> None:
    clients = []

    class FakeResources:
        __slots__ = ("kwargs",)

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeImage:
        @staticmethod
        def base(image):
            return f"base:{image}"

    class FakeParams:
        __slots__ = ("__dict__",)

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeProcess:
        async def exec(self, command, timeout):
            return SimpleNamespace(exit_code=7, result=f"ran:{command}:{timeout}")

    class FakeSandbox:
        id = "sandbox-1"

        def __init__(self):
            self.process = FakeProcess()
            self.deleted = 0

        async def delete(self):
            self.deleted += 1

    class FakeClient:
        def __init__(self):
            self.sandbox = FakeSandbox()
            self.params = None
            self.timeout = None
            self.closed = 0
            clients.append(self)

        async def create(self, params, timeout):
            self.params = params
            self.timeout = timeout
            return self.sandbox

        async def close(self):
            self.closed += 1

    monkeypatch.setattr(
        daytona_backend,
        "_load_daytona_sdk",
        lambda: {
            "AsyncDaytona": FakeClient,
            "CreateSandboxFromImageParams": FakeParams,
            "Image": FakeImage,
            "Resources": FakeResources,
        },
    )

    async def run():
        client = DaytonaSandboxClient.Config(
            create_timeout_s=12,
            auto_stop_interval_mins=5,
        ).build()
        session = await client.create(
            SandboxSpec(
                image="registry/task:latest",
                num_cpus=2,
                memory_mb=1536,
                storage_mb=2500,
                timeout_s=9,
                env={"TASK": "one"},
            ),
            owner_id="rollout-1",
        )
        result = await session.exec("echo hello", cwd="/app")
        await session.close()
        await session.close()
        return result

    result = asyncio.run(run())
    (client,) = clients
    assert client.timeout == 12
    assert client.params.image == "base:registry/task:latest"
    assert client.params.resources.kwargs == {"cpu": 2, "memory": 2, "disk": 3}
    assert client.params.network_block_all is False
    assert client.params.env_vars == {"TASK": "one"}
    assert client.params.labels["torchtitan.owner_id"] == "rollout-1"
    assert result.exit_code == 7
    assert "cd /app" in result.stdout
    assert client.sandbox.deleted == 1
    assert client.closed == 1
