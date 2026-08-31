# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rollouter backed by a Verifiers environment service."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from typing import Any, TYPE_CHECKING

from torchtitan.config import Configurable
from torchtitan.experiments.rl.examples.verifiers.components.dataset import (
    VerifiersTaskSample,
)
from torchtitan.experiments.rl.examples.verifiers.components.env_server import (
    VerifiersEnvServer,
)
from torchtitan.experiments.rl.rollout.advantage import AdvantageEstimator
from torchtitan.experiments.rl.rollout.rollouter import Rollouter, RolloutWorker
from torchtitan.experiments.rl.rollout.types import (
    GenerateFn,
    Rollout,
    RolloutGroup,
    RolloutStatus,
    RolloutTurn,
)
from torchtitan.experiments.rl.rubrics import RewardFn, Rubric
from torchtitan.experiments.rl.types import RolloutTurnID

if TYPE_CHECKING:
    from torchtitan.experiments.rl.actors.generator import SamplingConfig
    from torchtitan.experiments.rl.examples.verifiers.components.model_adapter import (
        GenerationMetadata,
        GeneratorModelAdapter,
    )
    from torchtitan.experiments.rl.renderer import RendererConfig


VERIFIERS_REWARD_KEY = "verifiers_reward"


class VerifiersRewardFn(RewardFn):
    """Return the reward produced by Verifiers."""

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        pass

    async def __call__(self, rollout: Rollout, env_input: object) -> float:
        del env_input
        for turn in reversed(rollout.turns):
            if VERIFIERS_REWARD_KEY in turn.env_rewards:
                return float(turn.env_rewards[VERIFIERS_REWARD_KEY])
        return 0.0


class VerifiersRollouter(Rollouter):
    """Run rollout groups through a locally managed Verifiers EnvServer."""

    @dataclass(kw_only=True, slots=True)
    class Config(Rollouter.Config):
        # Unused because this class replaces the base rollout-worker execution path.
        worker: RolloutWorker.Config | None = None
        # Configuration for the locally managed Verifiers environment server.
        env_server: VerifiersEnvServer.Config
        # TorchTitan rubric that consumes rewards returned by Verifiers.
        rubric: Rubric.Config
        # Converts sibling rollout rewards into training advantages.
        advantage: Configurable.Config = field(
            default_factory=AdvantageEstimator.Config
        )

        # Interface on which the local HTTP model adapter listens.
        model_adapter_bind_host: str = "127.0.0.1"
        # Adapter port; zero requests an ephemeral port from the operating system.
        model_adapter_bind_port: int = 0
        # Base URL given to the Verifiers training client after port substitution.
        model_adapter_base_url: str = "http://127.0.0.1:{port}/v1"
        # Maximum concurrent rollouts sharing one renderer instance.
        renderer_multiplex: int = 256
        # Context limit advertised by the local model adapter.
        max_model_len: int
        # Maximum time to wait for the Verifiers server to become healthy.
        connection_timeout_sec: float = 120.0

        def __post_init__(self) -> None:
            Rollouter.Config.__post_init__(self)
            if not 0 <= self.model_adapter_bind_port <= 65535:
                raise ValueError("model_adapter_bind_port must be between 0 and 65535")
            if (
                self.model_adapter_bind_port == 0
                and "{port}" not in self.model_adapter_base_url
            ):
                raise ValueError(
                    "model_adapter_base_url must contain '{port}' when binding "
                    "an ephemeral port"
                )
            if self.renderer_multiplex <= 0:
                raise ValueError("renderer_multiplex must be positive")
            if self.max_model_len <= 0:
                raise ValueError("max_model_len must be positive")
            if self.connection_timeout_sec <= 0:
                raise ValueError("connection_timeout_sec must be positive")

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._verifiers_config = config
        self._rubric: Rubric = config.rubric.build()
        self._advantage_estimator: AdvantageEstimator = config.advantage.build()
        self._env_server = config.env_server.build()
        self._adapter: GeneratorModelAdapter | None = None
        self._env_client: Any = None
        self._train_client_config: Any = None

    async def setup_async(
        self,
        *,
        renderer_config: RendererConfig,
        hf_assets_path: str,
    ) -> None:
        """Start the EnvServer and connect it to TorchTitan generation."""
        if self._env_client is not None:
            return

        from verifiers.v1.configs.client import TrainClientConfig
        from verifiers.v1.serve.client import EnvClient

        from torchtitan.experiments.rl.examples.verifiers.components.model_adapter import (
            GeneratorModelAdapter,
        )

        adapter = GeneratorModelAdapter(
            host=self._verifiers_config.model_adapter_bind_host,
            port=self._verifiers_config.model_adapter_bind_port,
            model=hf_assets_path,
            max_model_len=self._verifiers_config.max_model_len,
        )
        server_address = await self._env_server.start()
        env_client = None
        try:
            await adapter.start()
            env_client = EnvClient(server_address)
            await env_client.wait_for_server_startup(
                timeout=self._verifiers_config.connection_timeout_sec
            )
            train_client_config = TrainClientConfig(
                base_url=self._verifiers_config.model_adapter_base_url.format(
                    port=adapter.port
                ),
                # No API key is needed. This intentionally unset variable makes
                # Verifiers use "EMPTY" instead of forwarding PRIME_API_KEY.
                api_key_var="TORCHTITAN_VERIFIERS_API_KEY",
                renderer=renderer_config.as_renderers_config(),
                multiplex=self._verifiers_config.renderer_multiplex,
                renderer_model_name=hf_assets_path,
            )
        except BaseException:
            if env_client is not None:
                await env_client.close()
            try:
                await adapter.close()
            finally:
                await self._env_server.close()
            raise
        self._adapter = adapter
        self._env_client = env_client
        self._train_client_config = train_client_config

    async def close(self) -> None:
        """Close the Verifiers client, model adapter, and environment server."""
        try:
            if self._env_client is not None:
                await self._env_client.close()
        finally:
            self._env_client = None
            self._train_client_config = None
            try:
                if self._adapter is not None:
                    await self._adapter.close()
            finally:
                self._adapter = None
                await self._env_server.close()

    async def run_group_rollouts(
        self,
        *,
        generate_fn: GenerateFn,
        sample: object,
        group_id: int,
        group_size: int,
        sampling: SamplingConfig,
    ) -> RolloutGroup:
        """Run sibling episodes through Verifiers, then compute advantages."""
        rollouts = await asyncio.gather(
            *(
                self._run_single_rollout(
                    generate_fn=generate_fn,
                    sample=sample,
                    sampling=(
                        sampling
                        if sampling.seed is None
                        else replace(sampling, seed=sampling.seed + rollout_id)
                    ),
                    group_id=group_id,
                    rollout_id=rollout_id,
                )
                for rollout_id in range(group_size)
            )
        )

        outputs = await self._rubric.score_group(rollouts, sample)
        for rollout, output in zip(rollouts, outputs, strict=True):
            rollout.reward = output.reward
            rollout.reward_breakdown = output.reward_breakdown

        group = RolloutGroup(group_id=group_id, rollouts=rollouts)
        advantages = self._advantage_estimator(group)
        for rollout, advantage in zip(group.rollouts, advantages, strict=True):
            rollout.advantage = advantage
        return group

    async def _run_single_rollout(
        self,
        *,
        generate_fn: GenerateFn,
        sample: object,
        sampling: SamplingConfig,
        group_id: int,
        rollout_id: int,
    ) -> Rollout:
        """Send one task to Verifiers and convert its trace to a rollout."""
        if not isinstance(sample, VerifiersTaskSample):
            raise TypeError("Verifiers requires a VerifiersTaskSample")
        if (
            self._adapter is None
            or self._env_client is None
            or self._train_client_config is None
        ):
            raise RuntimeError("Verifiers rollouter is not initialized")

        from verifiers.v1.types import SamplingConfig as VerifiersSamplingConfig

        self._adapter.set_generate_fn(generate_fn)
        episode = await self._env_client.run(
            task_data=sample.task_data,
            client=self._train_client_config,
            model=self._adapter.model,
            sampling=VerifiersSamplingConfig(
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                max_tokens=sampling.max_tokens,
                seed=sampling.seed,
            ),
        )
        traces = [trace for trace in episode.traces if trace.agent.trainable]
        if len(traces) != 1:
            raise ValueError(
                "Verifiers expects one trainable trace per episode; got "
                f"{len(traces)}"
            )
        trace = traces[0]
        generation_metadata = self._adapter.pop_generation_metadata(trace.id)
        turns = self.trace_to_rollout_turns(
            trace=trace,
            generation_metadata=generation_metadata,
            group_id=group_id,
            rollout_id=rollout_id,
        )
        status = self.rollout_status(episode=episode, trace=trace)
        if not turns:
            status = RolloutStatus.ERROR
        else:
            turns[-1].env_rewards[VERIFIERS_REWARD_KEY] = trace.reward
        return Rollout(
            group_id=group_id,
            rollout_id=rollout_id,
            status=status,
            turns=turns,
        )

    @staticmethod
    def rollout_status(*, episode: Any, trace: Any) -> RolloutStatus:
        if not episode.ok or not trace.ok:
            return RolloutStatus.ERROR
        if not trace.is_truncated:
            return RolloutStatus.COMPLETED
        if trace.stop_condition == "max_turns":
            return RolloutStatus.TRUNCATED_MAX_TURNS
        return RolloutStatus.TRUNCATED_LENGTH

    @staticmethod
    def trace_to_rollout_turns(
        *,
        trace: Any,
        generation_metadata: list[GenerationMetadata],
        group_id: int,
        rollout_id: int,
    ) -> list[RolloutTurn]:
        successful_calls = [call for call in trace.calls if call.node is not None]
        if len(successful_calls) != len(generation_metadata):
            raise ValueError(
                "Verifiers trace/model-adapter call count mismatch: "
                f"trace={len(successful_calls)}, adapter={len(generation_metadata)}"
            )
        metadata_by_node = {
            call.node: call_metadata
            for call, call_metadata in zip(
                successful_calls, generation_metadata, strict=True
            )
        }
        node_index = {id(node): index for index, node in enumerate(trace.nodes)}
        trained_nodes: set[int] = set()
        turns: list[RolloutTurn] = []

        for branch in trace.branches:
            token_ids = branch.token_ids
            logprobs = branch.logprobs
            branch_offset = 0
            for node in branch.nodes:
                index = node_index[id(node)]
                mask = list(node.mask)
                if node.sampled and any(mask):
                    if index in trained_nodes:
                        mask = [False] * len(mask)
                    else:
                        trained_nodes.add(index)
                for start, end in _trainable_token_spans(mask):
                    call_metadata = metadata_by_node.get(index)
                    if call_metadata is None:
                        raise ValueError(
                            f"sampled Verifiers node {index} has no generation metadata"
                        )
                    absolute_start = branch_offset + start
                    absolute_end = branch_offset + end
                    turns.append(
                        RolloutTurn(
                            rollout_id=RolloutTurnID(
                                group_id=group_id,
                                rollout_id=rollout_id,
                                turn_id=len(turns),
                            ),
                            prompt_token_ids=list(token_ids[:absolute_start]),
                            completion_token_ids=list(
                                token_ids[absolute_start:absolute_end]
                            ),
                            completion_logprobs=list(
                                logprobs[absolute_start:absolute_end]
                            ),
                            min_policy_version=call_metadata.min_policy_version,
                            max_policy_version=call_metadata.max_policy_version,
                            metrics=list(call_metadata.metrics),
                        )
                    )
                branch_offset += len(node.token_ids)
        return turns


def _trainable_token_spans(mask: list[bool]) -> list[tuple[int, int]]:
    """Return half-open token spans marked trainable by a Verifiers node mask."""
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, sampled in enumerate([*mask, False]):
        if sampled and start is None:
            start = index
        elif not sampled and start is not None:
            spans.append((start, index))
            start = None
    return spans
