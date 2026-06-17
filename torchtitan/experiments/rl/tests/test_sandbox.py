# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the shared sandbox layer's provisioning policy.

No containers: a fake backend (``_FlakyFactory``) exercises the base
``SandboxFactory`` retry / fail-fast / concurrency-cap behavior in process.
"""

import asyncio
from dataclasses import dataclass

import pytest

from torchtitan.experiments.rl.sandbox import (
    ExecResult,
    SandboxFactory,
    SweProvisionError,
    SweProvisionUserError,
)


class _FakeSandbox:
    """Minimal in-process Sandbox handle."""

    def __init__(self) -> None:
        self.closed = False

    async def exec(self, cmd: str, *, timeout_s: float) -> ExecResult:
        return ExecResult(output="", exit_code=0)

    async def write_file(self, path: str, content: str) -> None:
        pass

    async def read_file(self, path: str) -> str:
        return ""

    async def close(self) -> None:
        self.closed = True


class _FlakyFactory(SandboxFactory):
    """Fake backend whose ``_provision_one`` fails a configurable number of times."""

    @dataclass(kw_only=True, slots=True)
    class Config(SandboxFactory.Config):
        fail_times: int = 0
        user_error: bool = False
        hold_s: float = 0.0

    def __init__(self, config: "Config") -> None:
        super().__init__(config)
        self.calls = 0
        self.in_flight = 0
        self.peak_in_flight = 0

    async def _provision_one(self, *, image: str):
        self.calls += 1
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            if self._config.user_error:
                raise SweProvisionUserError("bad image")
            if self.calls <= self._config.fail_times:
                raise SweProvisionError("transient")
            if self._config.hold_s:
                await asyncio.sleep(self._config.hold_s)
            return _FakeSandbox()
        finally:
            self.in_flight -= 1


def _cfg(**kw):
    kw.setdefault("provision_retry_delay_s", 0.0)
    return _FlakyFactory.Config(**kw)


def test_provision_succeeds_first_try():
    f = _FlakyFactory(_cfg())
    sb = asyncio.run(f.provision(image="img"))
    assert isinstance(sb, _FakeSandbox)
    assert f.calls == 1


def test_provision_retries_then_succeeds():
    f = _FlakyFactory(_cfg(fail_times=2, provision_retries=3))
    sb = asyncio.run(f.provision(image="img"))
    assert isinstance(sb, _FakeSandbox)
    assert f.calls == 3  # two transient failures then success


def test_provision_exhausts_retries():
    f = _FlakyFactory(_cfg(fail_times=99, provision_retries=3))
    with pytest.raises(SweProvisionError):
        asyncio.run(f.provision(image="img"))
    assert f.calls == 3


def test_provision_user_error_fails_fast():
    f = _FlakyFactory(_cfg(user_error=True, provision_retries=3))
    with pytest.raises(SweProvisionUserError):
        asyncio.run(f.provision(image="img"))
    assert f.calls == 1  # no retries on a non-retryable bad input


def test_provision_respects_concurrency_cap():
    f = _FlakyFactory(_cfg(max_concurrent_provision=2, hold_s=0.05))

    async def run_many():
        await asyncio.gather(*(f.provision(image=f"img{i}") for i in range(6)))

    asyncio.run(run_many())
    assert f.calls == 6
    assert f.peak_in_flight <= 2  # the semaphore capped concurrent provisions


def test_repo_root_exposed():
    f = _FlakyFactory(_cfg(repo_root="/work"))
    assert f.repo_root == "/work"
