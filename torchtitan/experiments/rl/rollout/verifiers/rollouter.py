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

from renderers import config_from_name, Renderer

from torchtitan.experiments.rl.environment import MessageEnv
from torchtitan.experiments.rl.rollout.rollouter import Rollouter
from torchtitan.experiments.rl.rollout.types import (
    GenerateFn,
    Rollout,
    RolloutGroup,
    RolloutStatus,
    RolloutTurn,
)
from torchtitan.experiments.rl.rollout.verifiers.dataset import (
    VerifiersTaskDataset,
    VerifiersTaskSample,
)
from torchtitan.experiments.rl.rubrics import RewardFn
from torchtitan.experiments.rl.types import RolloutTurnID, SamplingConfig

if TYPE_CHECKING:
    from torchtitan.experiments.rl.rollout.verifiers.model_adapter import (
        GenerationEvidence,
    )

VERIFIERS_REWARD_KEY = "verifiers_reward"


class VerifiersRewardFn(RewardFn):
    """Return the reward produced by the Verifiers environment service."""

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
    """Run rollout groups through a Verifiers EnvServer."""

    @dataclass(kw_only=True, slots=True)
    class Config(Rollouter.Config):
        train_dataset: VerifiersTaskDataset.Config
        validation_dataset: VerifiersTaskDataset.Config
        message_env: MessageEnv.Config | None = None

        model_adapter_bind_host: str = "0.0.0.0"
        model_adapter_bind_port: int = 8001
        model_adapter_base_url: str = "http://127.0.0.1:{port}/v1"
        """OpenAI base URL reachable from the Verifiers environment worker."""

        model_name: str = "Qwen/Qwen3-8B"
        renderer_model_name: str = "Qwen/Qwen3-8B"
        renderer_name: str = "qwen3"
        renderer_kwargs: dict[str, Any] = field(
            default_factory=lambda: {
                "enable_thinking": True,
                "thinking_retention": "all",
            }
        )
        renderer_multiplex: int = 256
        """Concurrent rollouts sharing one lazily created renderer."""
        max_model_len: int = 32768
        connection_timeout_sec: float = 120.0

        def __post_init__(self) -> None:
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
        from torchtitan.experiments.rl.rollout.verifiers.model_adapter import (
            GeneratorModelAdapter,
        )

        super().__init__(config)
        self._config = config
        self._adapter = GeneratorModelAdapter(
            host=config.model_adapter_bind_host,
            port=config.model_adapter_bind_port,
            model=config.model_name,
            max_model_len=config.max_model_len,
        )
        self._env_client: Any = None
        self._train_client_config: Any = None

    async def connect_verifiers_env_server(
        self, verifiers_env_server_address: str
    ) -> None:
        """Connect to the EnvServer and validate configured task indices."""
        if self._env_client is not None:
            return

        from verifiers.v1.configs.client import TrainClientConfig
        from verifiers.v1.serve.client import EnvClient

        await self._adapter.start()
        env_client = EnvClient(verifiers_env_server_address)
        try:
            await env_client.wait_for_server_startup(
                timeout=self._config.connection_timeout_sec
            )
            renderer = config_from_name(self._config.renderer_name)
            if renderer is None:
                raise ValueError("renderer_name must resolve to a concrete renderer")
            renderer = type(renderer)(**self._config.renderer_kwargs)
            train_client_config = TrainClientConfig(
                base_url=self._config.model_adapter_base_url.format(
                    port=self._adapter.port
                ),
                api_key_var="TORCHTITAN_VERIFIERS_API_KEY",
                renderer=renderer,
                multiplex=self._config.renderer_multiplex,
                renderer_model_name=self._config.renderer_model_name,
            )
        except BaseException:
            await env_client.close()
            await self._adapter.close()
            raise
        self._env_client = env_client
        self._train_client_config = train_client_config

    async def close(self) -> None:
        """Close the EnvClient and model adapter."""
        try:
            if self._env_client is not None:
                await self._env_client.close()
        finally:
            self._env_client = None
            self._train_client_config = None
            await self._adapter.close()

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
        """Run sibling episodes through EnvServer, then score the group."""
        del renderer
        rollouts = await asyncio.gather(
            *(
                self.run_single_rollout(
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
        outputs = await self.score_group(rollouts, sample)
        for rollout, output in zip(rollouts, outputs, strict=True):
            rollout.reward = output.reward
            rollout.reward_breakdown = output.reward_breakdown

        group = RolloutGroup(group_id=group_id, rollouts=rollouts)
        advantages = self.advantage_estimator(group)
        for rollout, advantage in zip(group.rollouts, advantages, strict=True):
            rollout.advantage = advantage
        return group

    async def run_single_rollout(
        self,
        *,
        generate_fn: GenerateFn,
        sample: object,
        sampling: SamplingConfig,
        group_id: int,
        rollout_id: int,
    ) -> Rollout:
        """Send one task to EnvServer and convert its trace to a rollout."""
        if not isinstance(sample, VerifiersTaskSample):
            raise TypeError("Verifiers requires a VerifiersTaskSample")
        if self._env_client is None or self._train_client_config is None:
            raise RuntimeError("Verifiers is not connected to an EnvServer")

        from verifiers.v1.types import SamplingConfig as VerifiersSamplingConfig

        self._adapter.set_generate_fn(generate_fn)
        episode = await self._env_client.run(
            task_data=sample.task_data,
            client=self._train_client_config,
            model=self._config.model_name,
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
        evidence = self._adapter.take_evidence(trace.id)
        turns = self.trace_to_rollout_turns(
            trace=trace,
            evidence=evidence,
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
        evidence: list[GenerationEvidence],
        group_id: int,
        rollout_id: int,
    ) -> list[RolloutTurn]:
        successful_calls = [call for call in trace.calls if call.node is not None]
        if len(successful_calls) != len(evidence):
            raise ValueError(
                "Verifiers trace/model-adapter call count mismatch: "
                f"trace={len(successful_calls)}, adapter={len(evidence)}"
            )
        evidence_by_node = {
            call.node: call_evidence
            for call, call_evidence in zip(successful_calls, evidence, strict=True)
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
                for start, end in _trainable_token_spans_from_mask(mask):
                    call_evidence = evidence_by_node.get(index)
                    if call_evidence is None:
                        raise ValueError(
                            f"sampled Verifiers node {index} has no generation evidence"
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
                            min_policy_version=call_evidence.min_policy_version,
                            max_policy_version=call_evidence.max_policy_version,
                            metrics=list(call_evidence.metrics),
                        )
                    )
                branch_offset += len(node.token_ids)
        return turns


def _trainable_token_spans_from_mask(
    mask: list[bool],
) -> list[tuple[int, int]]:
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
