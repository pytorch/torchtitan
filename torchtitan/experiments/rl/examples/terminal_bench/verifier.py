# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import shlex
from dataclasses import dataclass

from torchtitan.config import Configurable
from torchtitan.experiments.rl.examples.terminal_bench.data import (
    TerminalBenchArtifact,
    TerminalBenchSample,
)
from torchtitan.experiments.rl.sandbox import SandboxClient


@dataclass(frozen=True, kw_only=True, slots=True)
class TerminalBenchVerifierResult:
    """Raw signals returned by the Terminal-Bench verifier."""

    reward: float
    exit_code: int

    def as_reward_signals(self) -> dict[str, float]:
        return {
            "terminal_bench": self.reward,
            "verifier_exit_code": float(self.exit_code),
        }


class TerminalBenchVerifier(Configurable):
    """Runs Terminal-Bench tests in a fresh verifier sandbox."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        command: str = "/bin/bash /tests/test.sh"
        reward_path: str = "/logs/verifier/reward.txt"

        def __post_init__(self) -> None:
            if not self.command:
                raise ValueError("TerminalBenchVerifier.command must not be empty")
            if not self.reward_path:
                raise ValueError("TerminalBenchVerifier.reward_path must not be empty")

    def __init__(
        self,
        config: Config,
        *,
        sandbox_client: SandboxClient,
    ) -> None:
        self._config = config
        self._sandbox_client = sandbox_client

    async def verify(
        self,
        *,
        sample: TerminalBenchSample,
        artifacts: tuple[TerminalBenchArtifact, ...],
        owner_id: str,
    ) -> TerminalBenchVerifierResult:
        """Import artifacts, run the verifier, and destroy its sandbox."""
        sandbox = await self._sandbox_client.create(
            sample.verifier_sandbox,
            owner_id=f"{owner_id}-verifier",
        )
        try:
            for artifact in artifacts:
                await sandbox.upload(artifact.local_path, artifact.remote_path)
            verifier_result = await sandbox.exec(
                self._config.command,
                timeout_s=sample.verifier_timeout_s,
            )
            reward_result = await sandbox.exec(
                f"cat {shlex.quote(self._config.reward_path)}",
                timeout_s=30,
            )
            if reward_result.exit_code != 0:
                raise RuntimeError(
                    "Terminal-Bench verifier did not produce a reward: "
                    f"{reward_result.stdout}{reward_result.stderr}"
                )
            try:
                reward = float(reward_result.stdout.strip())
            except ValueError as error:
                raise RuntimeError(
                    "Terminal-Bench verifier reward is not numeric: "
                    f"{reward_result.stdout!r}"
                ) from error
            return TerminalBenchVerifierResult(
                reward=reward,
                exit_code=verifier_result.exit_code,
            )
        finally:
            await sandbox.close()
