# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rollouter backed by a Verifiers environment service."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from typing import Annotated, Any, TYPE_CHECKING

import tyro
from verifiers.v1.configs.client import TrainClientConfig as VerifiersTrainClientConfig
from verifiers.v1.configs.taskset import TasksetConfig as VerifiersTasksetConfig
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.serve.client import EnvClient as VerifiersEnvClient
from verifiers.v1.types import SamplingConfig as VerifiersSamplingConfig

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.experiments.rl.examples.verifiers.components.data import (
    VerifiersTaskDataset,
    VerifiersTaskSample,
)
from torchtitan.experiments.rl.examples.verifiers.components.env_server import (
    VerifiersEnvServer,
)
from torchtitan.experiments.rl.examples.verifiers.components.generation_server import (
    GenerationServer,
    VerifiersGenerationMetadata,
)
from torchtitan.experiments.rl.renderer import RendererConfig, RenderersLibraryConfig
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


VERIFIERS_REWARD_KEY = "verifiers_reward"


class RewardFromVerifiers(RewardFn):
    """Pass through the reward that Verifiers computed during the rollout."""

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
    """Adapt Verifiers execution to TitanRL's ``Rollouter`` contract.

    Verifiers owns environment execution, including its process pool and the
    message/tool loop, so this path bypasses TitanRL's ``RolloutWorkerActor``,
    ``RolloutWorker``, ``MessageEnv``, and ``TokenEnv``. Token-in/token-out is
    preserved through ``GenerationServer``, which forwards Verifiers model
    requests to TitanRL's controller-provided ``GenerateFn``.

    TitanRL still owns dataset and group scheduling, generator routing, policy
    metadata, conversion to ``RolloutTurn``, reward and advantage assignment,
    and the downstream training path.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Rollouter.Config):
        train_dataset: VerifiersTaskDataset.Config
        """Verifiers taskset used for training samples and environment scoring."""

        validation_dataset: VerifiersTaskDataset.Config
        """Verifiers taskset used for validation samples."""

        worker: Annotated[RolloutWorker.Config | None, tyro.conf.Suppress] = None
        """Inherited schema field; hidden because Verifiers replaces that path."""

        worker_pool_size: Annotated[int, tyro.conf.Suppress] = 1
        """Unused because the Verifiers EnvServer owns its worker pool."""

        num_threads_per_worker: Annotated[int, tyro.conf.Suppress] = 1
        """Unused because no TitanRL rollout-worker process is spawned."""

        verifiers_env_server: VerifiersEnvServer.Config
        """Environment server spawned and owned by this rollouter."""

        rubric: Rubric.Config
        """TorchTitan rubric that consumes rewards returned by Verifiers."""

        advantage: AdvantageEstimator.Config = field(
            default_factory=AdvantageEstimator.Config
        )
        """TitanRL estimator applied after Verifiers returns rollout rewards."""

        generation_server: GenerationServer.Config = field(
            default_factory=GenerationServer.Config
        )
        """Local HTTP bridge from Verifiers to TitanRL generation."""

        renderer_multiplex: int = 256
        """Maximum concurrent rollouts sharing one Verifiers renderer instance.

        TODO: evaluate sharing this renderer pool with TitanRL's native rollout
        path instead of keeping it specific to the Verifiers client.
        """

        connection_timeout_sec: float = 120.0
        """Maximum time to wait for the Verifiers server to become healthy."""

        def __post_init__(self) -> None:
            Rollouter.Config.__post_init__(self)
            configured_taskset = self.verifiers_env_server.environment.taskset
            if configured_taskset not in (
                VerifiersTasksetConfig(),
                self.train_dataset.verifiers_taskset,
            ):
                raise ValueError(
                    "verifiers_env_server.environment.taskset is derived from "
                    "train_dataset.verifiers_taskset and must not configure a "
                    "different taskset"
                )
            self.verifiers_env_server = replace(
                self.verifiers_env_server,
                environment=self.verifiers_env_server.environment.model_copy(
                    update={"taskset": self.train_dataset.verifiers_taskset}
                ),
                local_taskset_module=_local_taskset_module(
                    self.train_dataset.verifiers_taskset
                ),
            )
            if self.renderer_multiplex <= 0:
                raise ValueError("renderer_multiplex must be positive")
            if self.connection_timeout_sec <= 0:
                raise ValueError("connection_timeout_sec must be positive")

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._verifiers_config = config
        self._rubric: Rubric = config.rubric.build()
        self._advantage_estimator: AdvantageEstimator = config.advantage.build()
        self._verifiers_env_server = config.verifiers_env_server.build()
        self._generation_server: GenerationServer | None = None
        self._verifiers_env_client: VerifiersEnvClient | None = None
        self._verifiers_train_client_config: VerifiersTrainClientConfig | None = None

    async def setup_async(
        self,
        *,
        tokenizer_config: HuggingFaceTokenizer.Config,
        renderer_config: RendererConfig,
        hf_assets_path: str,
    ) -> None:
        """Start the EnvServer and connect it to TorchTitan generation."""
        del tokenizer_config
        if self._verifiers_env_client is not None:
            return
        if not isinstance(renderer_config, RenderersLibraryConfig):
            raise ValueError(
                "Verifiers requires RenderersLibraryConfig so its client can "
                "construct the same renderer in the environment-server process"
            )

        # Verifiers generates through an HTTP endpoint, while TorchTitan exposes
        # an in-process GenerateFn. This server bridges those interfaces.
        # TODO: remove this bridge when the controller can provide a generator
        # HTTP endpoint directly.
        generation_server = self._verifiers_config.generation_server.build()
        verifiers_server_address = await self._verifiers_env_server.start()
        verifiers_env_client = None
        try:
            await generation_server.start()
            verifiers_env_client = VerifiersEnvClient(verifiers_server_address)
            await verifiers_env_client.wait_for_server_startup(
                timeout=self._verifiers_config.connection_timeout_sec
            )
            verifiers_train_client_config = VerifiersTrainClientConfig(
                base_url=generation_server.base_url,
                # The local endpoint does not authenticate. An empty environment
                # variable name makes Verifiers send its required "EMPTY" value.
                api_key_var="",
                renderer=renderer_config.renderers_config,
                multiplex=self._verifiers_config.renderer_multiplex,
                renderer_model_name=hf_assets_path,
            )
        except BaseException:
            if verifiers_env_client is not None:
                await verifiers_env_client.close()
            try:
                await generation_server.close()
            finally:
                await self._verifiers_env_server.close()
            raise
        self._generation_server = generation_server
        self._verifiers_env_client = verifiers_env_client
        self._verifiers_train_client_config = verifiers_train_client_config

    async def close(self) -> None:
        """Close the Verifiers client, generation server, and environment server."""
        try:
            if self._verifiers_env_client is not None:
                await self._verifiers_env_client.close()
        finally:
            self._verifiers_env_client = None
            self._verifiers_train_client_config = None
            try:
                if self._generation_server is not None:
                    await self._generation_server.close()
            finally:
                self._generation_server = None
                await self._verifiers_env_server.close()

    async def run_group_rollouts(
        self,
        *,
        generate_fn: GenerateFn,
        sample: object,
        group_id: int,
        group_size: int,
        sampling: SamplingConfig,
    ) -> RolloutGroup:
        """Run sibling rollouts through Verifiers, then compute advantages."""
        if self._generation_server is None:
            raise RuntimeError("Verifiers rollouter is not initialized")
        self._generation_server.set_generate_fn(generate_fn)
        rollouts = await asyncio.gather(
            *(
                self._run_single_rollout(
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
        sample: object,
        sampling: SamplingConfig,
        group_id: int,
        rollout_id: int,
    ) -> Rollout:
        """Send one task to Verifiers and convert its trace to a rollout."""
        if not isinstance(sample, VerifiersTaskSample):
            raise TypeError("Verifiers requires a VerifiersTaskSample")
        if (
            self._generation_server is None
            or self._verifiers_env_client is None
            or self._verifiers_train_client_config is None
        ):
            raise RuntimeError("Verifiers rollouter is not initialized")

        # One VerifiersEnvClient.run executes a complete rollout. The harness
        # owns the multi-turn model/tool loop and calls the generation server once per
        # generation; run returns only after the rollout stops.
        verifiers_episode = await self._verifiers_env_client.run(
            task_data=sample.verifiers_task_data,
            client=self._verifiers_train_client_config,
            model=self._generation_server.model_id,
            sampling=VerifiersSamplingConfig(
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                max_tokens=sampling.max_tokens,
                seed=sampling.seed,
            ),
        )
        traces = [trace for trace in verifiers_episode.traces if trace.agent.trainable]
        if len(traces) != 1:
            raise ValueError(
                "Verifiers expects one trainable trace per rollout; got "
                f"{len(traces)}"
            )

        # Convert Verifiers' graph trace to TitanRL's linear rollout turns.
        trace = traces[0]
        generation_metadata = self._generation_server.pop_generation_metadata(trace.id)
        turns = self.trace_to_rollout_turns(
            trace=trace,
            generation_metadata=generation_metadata,
            group_id=group_id,
            rollout_id=rollout_id,
        )
        status = self.rollout_status(verifiers_episode=verifiers_episode, trace=trace)
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
    def rollout_status(*, verifiers_episode: Any, trace: Any) -> RolloutStatus:
        if not verifiers_episode.ok or not trace.ok:
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
        generation_metadata: VerifiersGenerationMetadata | None,
        group_id: int,
        rollout_id: int,
    ) -> list[RolloutTurn]:
        """Flatten a Verifiers trace into TorchTitan's trainable rollout turns.

        For example, a Verifiers branch containing user tokens ``[1, 2]``,
        sampled assistant tokens ``[3, 4]``, tool-result tokens ``[5, 6]``, and
        sampled assistant tokens ``[7, 8]`` becomes two TitanRL turns. Their
        prompts are ``[1, 2]`` and ``[1, 2, 3, 4, 5, 6]``; their completions are
        ``[3, 4]`` and ``[7, 8]``. Shared sampled graph nodes are emitted once.

        Verifiers does not return TorchTitan policy metadata, so every emitted
        turn receives the conservative min/max policy-version span accumulated
        by the generation server for the whole rollout. Generator metrics are
        attached once to avoid double counting.
        """
        if generation_metadata is None:
            if any(any(node.mask) for node in trace.nodes):
                raise ValueError(
                    "Verifiers trace has trainable tokens but no generation metadata"
                )
            return []

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
                            min_policy_version=generation_metadata.min_policy_version,
                            max_policy_version=generation_metadata.max_policy_version,
                            completion_message=message_to_wire(node.message),
                            metrics=(
                                list(generation_metadata.metrics) if not turns else []
                            ),
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


def _local_taskset_module(taskset: VerifiersTasksetConfig) -> str | None:
    """Return the module backing a locally registered taskset alias."""
    module = type(taskset).__module__
    alias = module.replace(".", "_").lower()
    return module if taskset.id == alias else None
