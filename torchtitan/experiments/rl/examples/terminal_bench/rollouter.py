# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import logging
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from renderers import Renderer

from torchtitan.experiments.rl.environment import TokenEnv
from torchtitan.experiments.rl.examples.terminal_bench.data import TerminalBenchDataset
from torchtitan.experiments.rl.examples.terminal_bench.env import TerminalBenchEnv
from torchtitan.experiments.rl.examples.terminal_bench.rubric import RewardTerminalBench
from torchtitan.experiments.rl.examples.terminal_bench.verifier import (
    TerminalBenchVerifier,
)
from torchtitan.experiments.rl.rollout import Rollout, RolloutGroup, RolloutStatus
from torchtitan.experiments.rl.rollout.rollouter import Rollouter
from torchtitan.experiments.rl.rollout.types import GenerateFn
from torchtitan.experiments.rl.rubrics import Rubric
from torchtitan.experiments.rl.sandbox import DockerSandboxClient, SandboxClient

if TYPE_CHECKING:
    from torchtitan.experiments.rl.actors.generator import SamplingConfig

logger = logging.getLogger(__name__)


class TerminalBenchRollouter(Rollouter):
    """Terminal-Bench rollouter using TorchTitan's framework-owned tool loop."""

    @dataclass(kw_only=True, slots=True)
    class Config(Rollouter.Config):
        train_dataset: TerminalBenchDataset.Config = field(
            default_factory=TerminalBenchDataset.Config
        )
        validation_dataset: TerminalBenchDataset.Config = field(
            default_factory=lambda: TerminalBenchDataset.Config(shuffle=False)
        )
        rubric: Rubric.Config = field(
            default_factory=lambda: Rubric.Config(
                reward_fns=[RewardTerminalBench.Config()],
                error_reward=0.0,
            )
        )
        message_env: TerminalBenchEnv.Config = field(
            default_factory=TerminalBenchEnv.Config
        )
        verifier: TerminalBenchVerifier.Config = field(
            default_factory=TerminalBenchVerifier.Config
        )
        token_env: TokenEnv.Config = field(
            default_factory=lambda: TokenEnv.Config(
                max_rollout_tokens=32_768,
                max_num_turns=64,
            )
        )
        sandbox_client: SandboxClient.Config = field(
            default_factory=DockerSandboxClient.Config
        )

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._sandbox_client: SandboxClient = config.sandbox_client.build()
        self._verifier = config.verifier.build(
            sandbox_client=self._sandbox_client,
        )

    def _make_env_pair(
        self,
        *,
        sample: object,
        renderer: Renderer,
    ) -> tuple[TerminalBenchEnv, TokenEnv]:
        message_env = self._message_env_config.build(
            env_input=sample,
            sandbox_client=self._sandbox_client,
        )
        token_env = self._token_env_config.build(
            message_env=message_env,
            renderer=renderer,
        )
        return message_env, token_env

    def make_env_group(
        self,
        *,
        sample: object,
        group_size: int,
        renderer: Renderer,
    ) -> list[TokenEnv]:
        return [
            self._make_env_pair(sample=sample, renderer=renderer)[1]
            for _ in range(group_size)
        ]

    async def run_group_rollouts(
        self,
        *,
        generate_fn: GenerateFn,
        sample: object,
        group_id: int,
        group_size: int,
        sampling: SamplingConfig,
        renderer: Renderer,
    ) -> RolloutGroup:
        """Run work sandboxes, then verify their exported artifacts."""
        env_pairs = [
            self._make_env_pair(sample=sample, renderer=renderer)
            for _ in range(group_size)
        ]
        try:
            rollouts = await asyncio.gather(
                *(
                    self._run_verified_rollout(
                        generate_fn=generate_fn,
                        sample=sample,
                        message_env=message_env,
                        token_env=token_env,
                        sampling=(
                            sampling
                            if sampling.seed is None
                            else replace(sampling, seed=sampling.seed + rollout_id)
                        ),
                        group_id=group_id,
                        rollout_id=rollout_id,
                    )
                    for rollout_id, (message_env, token_env) in enumerate(env_pairs)
                )
            )
        finally:
            await asyncio.gather(
                *(token_env.close() for _, token_env in env_pairs),
                return_exceptions=True,
            )

        outputs = await self.score_group(rollouts, sample)
        for rollout, output in zip(rollouts, outputs, strict=True):
            rollout.reward = output.reward
            rollout.reward_breakdown = output.reward_breakdown

        group = RolloutGroup(group_id=group_id, rollouts=rollouts)
        advantages = self.advantage_estimator(group)
        for rollout, advantage in zip(group.rollouts, advantages, strict=True):
            rollout.advantage = advantage
        return group

    async def _run_verified_rollout(
        self,
        *,
        generate_fn: GenerateFn,
        sample: object,
        message_env: TerminalBenchEnv,
        token_env: TokenEnv,
        sampling: SamplingConfig,
        group_id: int,
        rollout_id: int,
    ) -> Rollout:
        rollout = await super()._run_single_rollout(
            generate_fn=generate_fn,
            env=token_env,
            sampling=sampling,
            group_id=group_id,
            rollout_id=rollout_id,
        )

        with tempfile.TemporaryDirectory(prefix="torchtitan-tb-artifacts-") as temp:
            try:
                artifacts = await message_env.export_artifacts(Path(temp))
            except Exception:
                logger.exception(
                    "failed to export Terminal-Bench artifacts for %s/%d",
                    group_id,
                    rollout_id,
                )
                rollout.status = RolloutStatus.ERROR
                return rollout

            try:
                await token_env.close()
            except Exception:
                logger.exception(
                    "failed to close Terminal-Bench work sandbox for %s/%d",
                    group_id,
                    rollout_id,
                )
                rollout.status = RolloutStatus.ERROR

            try:
                verifier_result = await self._verifier.verify(
                    sample=sample,
                    artifacts=artifacts,
                    owner_id=message_env.owner_id,
                )
            except Exception:
                logger.exception(
                    "Terminal-Bench verifier failed for %s/%d",
                    group_id,
                    rollout_id,
                )
                rollout.status = RolloutStatus.ERROR
                return rollout

        if not rollout.turns:
            rollout.status = RolloutStatus.ERROR
            return rollout
        rollout.turns[-1].env_rewards.update(verifier_result.as_reward_signals())
        return rollout
