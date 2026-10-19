# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rollouter backed by a Verifiers environment service."""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field, replace
from typing import Any, TYPE_CHECKING

import numpy as np
from verifiers.v1.configs.client import TrainClientConfig as VerifiersTrainClientConfig
from verifiers.v1.configs.taskset import TasksetConfig as VerifiersTasksetConfig
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.errors import (
    HarnessError as VerifiersHarnessError,
    ProviderError as VerifiersProviderError,
)
from verifiers.v1.serve.client import EnvClient as VerifiersEnvClient
from verifiers.v1.types import SamplingConfig as VerifiersSamplingConfig

from torchtitan.components.renderer import RendererConfig, RenderersConfigAdapter

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.rl.examples.verifiers.data import (
    VerifiersTaskDataset,
    VerifiersTaskSample,
)
from torchtitan.rl.examples.verifiers.env_server import VerifiersEnvServer
from torchtitan.rl.generator.generation_server import (
    GenerationServer,
    GROUP_ID_SAMPLING_PARAM,
    METRICS_PREFIX_SAMPLING_PARAM,
    VerifiersGenerationMetadata,
)
from torchtitan.rl.generator.http_client import GeneratorClient
from torchtitan.rl.rollout.advantage import AdvantageEstimator
from torchtitan.rl.rollout.data import IterableRLDataLoader
from torchtitan.rl.rollout.message_env import MessageEnv
from torchtitan.rl.rollout.rollouter import (
    gather_rollouts,
    Rollouter,
    rollouts_from_gather,
    RolloutWorker,
)
from torchtitan.rl.rollout.rubric import RewardFn, Rubric
from torchtitan.rl.rollout.types import (
    GenerateFn,
    ReleaseSessionFn,
    Rollout,
    RolloutGroup,
    RolloutStatus,
    RolloutTurn,
    RolloutTurnID,
    split_prompt,
)

if TYPE_CHECKING:
    from torchtitan.rl.actors.system import ActorSystem
    from torchtitan.rl.generator.types import SamplingConfig


logger = logging.getLogger(__name__)

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


@dataclass(kw_only=True, slots=True)
class VerifiersValidationSet:
    """An extra validation set with its own taskset, hence its own EnvServer (e.g. a benchmark
    whose environment, images or grader differ from the training taskset's)."""

    dataset: VerifiersTaskDataset.Config
    """Verifiers taskset used for this set's validation samples."""

    verifiers_env_server: VerifiersEnvServer.Config
    """EnvServer for this set; its taskset is derived from `dataset`."""

    def __post_init__(self) -> None:
        configured_taskset = self.verifiers_env_server.environment.taskset
        if configured_taskset not in (
            VerifiersTasksetConfig(),
            self.dataset.verifiers_taskset,
        ):
            raise ValueError(
                "an extra validation set's verifiers_env_server.environment.taskset is derived "
                "from its dataset.verifiers_taskset and must not configure a different taskset"
            )
        self.verifiers_env_server = replace(
            self.verifiers_env_server,
            environment=self.verifiers_env_server.environment.model_copy(
                update={"taskset": self.dataset.verifiers_taskset}
            ),
            local_taskset_module=_local_taskset_module(self.dataset.verifiers_taskset),
        )


class VerifiersRollouter(Rollouter):
    """Adapt Verifiers execution to TitanRL's ``Rollouter`` contract.

    Verifiers owns environment execution, including its process pool and the
    message/tool loop, so this path bypasses TitanRL's `MessageEnv` and
    `TokenEnv`. The rollouter starts one Verifiers EnvServer; each rollout worker
    process runs a `VerifiersRolloutWorker` that connects to it, so the controller
    receives finished groups only, as with native rollouts. Token-in/token-out is
    preserved because Verifiers' agents post every model call straight to one
    generator's `GenerationServer`, which the router picks for each rollout.

    TitanRL still owns dataset and group scheduling, generator routing, policy
    metadata, conversion to ``RolloutTurn``, reward and advantage assignment,
    and the downstream training path.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Rollouter.Config):
        training_dataloader: IterableRLDataLoader.Config
        """Verifiers taskset used for training samples and environment scoring."""

        validation_dataset: VerifiersTaskDataset.Config
        """Verifiers taskset used for validation samples."""

        extra_validation_sets: dict[str, VerifiersValidationSet] = field(
            default_factory=dict
        )
        """Named validation sets on their own EnvServers, run by `AsyncLoopConfig.extra_validations`
        entries of the same name. Unlike `validation_dataset`, their taskset may differ from the
        training one."""

        worker: RolloutWorker.Config | None = None
        """Must stay None: the rollouter builds a `VerifiersRolloutWorker.Config` at setup,
        once the EnvServer's address is known."""

        verifiers_env_server: VerifiersEnvServer.Config
        """Environment server spawned and owned by this rollouter."""

        rubric: Rubric.Config
        """TorchTitan rubric that consumes rewards returned by Verifiers."""

        advantage: AdvantageEstimator.Config = field(
            default_factory=AdvantageEstimator.Config
        )
        """TitanRL estimator applied after Verifiers returns rollout rewards."""

        renderer_multiplex: int = 256
        """Maximum concurrent rollouts sharing one Verifiers renderer instance.

        TODO: evaluate sharing this renderer pool with TitanRL's native rollout
        path instead of keeping it specific to the Verifiers client.
        """

        connection_timeout_sec: float = 120.0
        """Maximum time to wait for the Verifiers server to become healthy."""

        def __post_init__(self) -> None:
            Rollouter.Config.__post_init__(self)
            if self.worker is not None:
                raise ValueError(
                    "VerifiersRollouter builds its worker config at setup; leave worker unset"
                )
            configured_taskset = self.verifiers_env_server.environment.taskset
            if configured_taskset not in (
                VerifiersTasksetConfig(),
                self.training_dataloader.dataset.verifiers_taskset,
            ):
                raise ValueError(
                    "verifiers_env_server.environment.taskset is derived from "
                    "training_dataloader.dataset.verifiers_taskset and must not configure a "
                    "different taskset"
                )
            self.verifiers_env_server = replace(
                self.verifiers_env_server,
                environment=self.verifiers_env_server.environment.model_copy(
                    update={
                        "taskset": self.training_dataloader.dataset.verifiers_taskset
                    }
                ),
                local_taskset_module=_local_taskset_module(
                    self.training_dataloader.dataset.verifiers_taskset
                ),
            )
            if self.renderer_multiplex <= 0:
                raise ValueError("renderer_multiplex must be positive")
            if self.connection_timeout_sec <= 0:
                raise ValueError("connection_timeout_sec must be positive")

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._verifiers_config = config
        self._verifiers_env_server = config.verifiers_env_server.build()
        self._extra_env_servers = {
            name: validation_set.verifiers_env_server.build()
            for name, validation_set in config.extra_validation_sets.items()
        }

    async def setup_async(
        self,
        *,
        actor_system: ActorSystem,
        tokenizer_config: HuggingFaceTokenizer.Config,
        renderer_config: RendererConfig,
        hf_assets_path: str,
    ) -> None:
        """Start the EnvServers, then the rollout workers that connect to them."""
        if not isinstance(renderer_config, RenderersConfigAdapter):
            raise ValueError(
                "Verifiers requires a renderer configured with from_renderers(...) "
                "so its client can construct the same renderer in the "
                "environment-server process"
            )
        config = self._verifiers_config
        try:
            env_server_address = await self._verifiers_env_server.start()
            extra_env_server_addresses = {
                name: await env_server.start()
                for name, env_server in self._extra_env_servers.items()
            }
            await super().setup_async(
                actor_system=actor_system,
                tokenizer_config=tokenizer_config,
                renderer_config=renderer_config,
                hf_assets_path=hf_assets_path,
                worker_config=VerifiersRolloutWorker.Config(
                    rubric=config.rubric,
                    advantage=config.advantage,
                    renderer_multiplex=config.renderer_multiplex,
                    connection_timeout_sec=config.connection_timeout_sec,
                    env_server_address=env_server_address,
                    extra_env_server_addresses=extra_env_server_addresses,
                ),
            )
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        """Stop the rollout workers, then the EnvServers."""
        try:
            await super().close()
        finally:
            for env_server in self._extra_env_servers.values():
                await env_server.close()
            await self._verifiers_env_server.close()

    def get_validation_samples(
        self, steps: int, validation_set: str | None = None
    ) -> list[object]:
        """Materialize one validation pass; an extra set's tasks are tagged with its name so they
        run on its EnvServer."""
        if validation_set is None:
            return Rollouter.get_validation_samples(self, steps)
        try:
            dataset = self._verifiers_config.extra_validation_sets[
                validation_set
            ].dataset
        except KeyError:
            raise ValueError(
                f"unknown validation set {validation_set!r}; extra_validation_sets has "
                f"{sorted(self._verifiers_config.extra_validation_sets)}"
            ) from None
        iterator = iter(dataset.build())
        samples = (
            list(iterator) if steps == -1 else list(itertools.islice(iterator, steps))
        )
        return [replace(sample, validation_set=validation_set) for sample in samples]


class VerifiersRolloutWorker(RolloutWorker):
    """Runs Verifiers rollout groups in a rollout worker process: the env client, the trace
    decode and the conversion to rollout turns.

    Unused inherited fields: `token_env`, and `rollout_timeout_s` (set Verifiers'
    `timeout.rollout`, which keeps the trace)."""

    @dataclass(kw_only=True, slots=True)
    class Config(RolloutWorker.Config):
        message_env: MessageEnv.Config | None = None
        """Unused: Verifiers runs the env and its tool loop."""

        renderer_multiplex: int
        """See `VerifiersRollouter.Config.renderer_multiplex`."""

        connection_timeout_sec: float
        """See `VerifiersRollouter.Config.connection_timeout_sec`."""

        env_server_address: str
        """Address of the rollouter's running EnvServer."""

        extra_env_server_addresses: dict[str, str] = field(default_factory=dict)
        """Address of each extra validation set's EnvServer, by set name."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._verifiers_config = config
        # Never closed: the worker process exits with the pool.
        self._verifiers_env_client: VerifiersEnvClient | None = None
        # Extra validation sets' EnvServer clients, by set name.
        self._extra_env_clients: dict[str, VerifiersEnvClient] = {}
        self._verifiers_train_client_config: VerifiersTrainClientConfig | None = None

    async def setup_async(
        self,
        *,
        tokenizer_config: HuggingFaceTokenizer.Config,
        renderer_config: RendererConfig,
        hf_assets_path: str,
    ) -> None:
        """Connect to the EnvServers. Verifiers renders in the env server, so no tokenizer here."""
        del tokenizer_config
        verifiers_env_client = VerifiersEnvClient(
            self._verifiers_config.env_server_address
        )
        extra_env_clients = {
            name: VerifiersEnvClient(address)
            for name, address in self._verifiers_config.extra_env_server_addresses.items()
        }
        try:
            for client in (verifiers_env_client, *extra_env_clients.values()):
                await client.wait_for_server_startup(
                    timeout=self._verifiers_config.connection_timeout_sec
                )
            verifiers_train_client_config = VerifiersTrainClientConfig(
                # Each rollout sets its generator's URL; see `_run_single_rollout`.
                base_url="",
                # The generator endpoint does not authenticate. An empty environment
                # variable name makes Verifiers send its required "EMPTY" value.
                api_key_var="",
                renderer=renderer_config.get_renderers_config(),
                multiplex=self._verifiers_config.renderer_multiplex,
                renderer_model_name=hf_assets_path,
            )
        except BaseException:
            for client in (verifiers_env_client, *extra_env_clients.values()):
                await client.close()
            raise
        self._verifiers_env_client = verifiers_env_client
        self._extra_env_clients = extra_env_clients
        self._verifiers_train_client_config = verifiers_train_client_config

    async def run_group(
        self,
        *,
        generate_fn: GenerateFn,
        release_session_fn: ReleaseSessionFn,
        sample: object,
        group_id: int,
        group_size: int,
        sampling: SamplingConfig,
    ) -> RolloutGroup:
        """Run sibling rollouts through Verifiers, then compute advantages."""
        # Each rollout releases its own generator sessions; see `_run_single_rollout`.
        del release_session_fn
        if not isinstance(generate_fn, GeneratorClient):
            raise TypeError(
                "Verifiers' agents call a generator over HTTP, so generate_fn must be a "
                f"GeneratorClient, got {type(generate_fn).__name__}"
            )
        # A cancel cancels every sibling, and with return_exceptions the gather still waits for each
        # one's lease release before it raises: the caller releases the group next.
        results, group_metrics = await gather_rollouts(
            self._run_single_rollout(
                generator_client=generate_fn,
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
        rollouts = rollouts_from_gather(results, group_id=group_id)

        scored = [
            rollout
            for rollout in rollouts
            if rollout.status is not RolloutStatus.ERROR_SYSTEM
        ]
        outputs = await self.score_group(scored, sample)
        for rollout, output in zip(scored, outputs, strict=True):
            rollout.reward = output.reward
            rollout.reward_breakdown = output.reward_breakdown

        advantages = self.advantage_estimator(
            RolloutGroup(group_id=group_id, rollouts=scored)
        )
        for rollout, advantage in zip(scored, advantages, strict=True):
            rollout.advantage = advantage
        return RolloutGroup(group_id=group_id, rollouts=rollouts, metrics=group_metrics)

    async def _run_single_rollout(
        self,
        *,
        generator_client: GeneratorClient,
        sample: object,
        sampling: SamplingConfig,
        group_id: int,
        rollout_id: int,
    ) -> Rollout:
        """Send one task to Verifiers, release its sessions, and convert its trace to a rollout.
        Replaces the base method with Verifiers' arguments; only this class's `run_group` calls it."""
        if not isinstance(sample, VerifiersTaskSample):
            raise TypeError("Verifiers requires a VerifiersTaskSample")
        if (
            self._verifiers_env_client is None
            or self._verifiers_train_client_config is None
        ):
            raise RuntimeError("Verifiers rollout worker is not initialized")

        # One VerifiersEnvClient.run executes a complete rollout. The harness owns the
        # multi-turn model/tool loop and posts every generation to one generator's server,
        # so all of the rollout's traces (sub-agents too) share that generator.
        endpoint = await generator_client.acquire_rollout(group_id=group_id)
        verifiers_episode = None
        try:
            env_client = (
                self._verifiers_env_client
                if sample.validation_set is None
                else self._extra_env_clients[sample.validation_set]
            )
            verifiers_episode = await env_client.run(
                task_data=sample.verifiers_task_data,
                # The same JSON for every rollout on a generator, so Verifiers keeps one HTTP
                # client per generator.
                client=self._verifiers_train_client_config.model_copy(
                    update={"base_url": f"{endpoint.url}/v1"}
                ),
                model=GenerationServer.model_id,
                sampling=VerifiersSamplingConfig(
                    temperature=sampling.temperature,
                    top_p=sampling.top_p,
                    max_tokens=sampling.max_tokens,
                    seed=sampling.seed,
                    num_topk_logprobs=sampling.num_topk_logprobs,
                    **{
                        GROUP_ID_SAMPLING_PARAM: group_id,
                        METRICS_PREFIX_SAMPLING_PARAM: generator_client.metrics_prefix,
                    },
                ),
            )
        finally:
            # An attempt Verifiers retried has no trace here; `release_groups` drops its
            # sessions at group end.
            generation_metadata_by_session = await generator_client.release_rollout(
                endpoint,
                group_id=group_id,
                session_ids=(
                    [trace.id for trace in verifiers_episode.traces]
                    if verifiers_episode is not None
                    else []
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
        generation_metadata = generation_metadata_by_session.get(trace.id)
        turns = self.trace_to_rollout_turns(
            trace=trace,
            generation_metadata=generation_metadata,
            group_id=group_id,
            rollout_id=rollout_id,
        )
        status = self.rollout_status(verifiers_episode=verifiers_episode, trace=trace)
        if not turns and status is not RolloutStatus.ERROR_SYSTEM:
            # Verifiers records an overlong first prompt as a failed model call; vLLM, the
            # renderer and the generation server all word it "maximum context length".
            first_prompt_too_long = any(
                call.error is not None
                and "maximum context length" in call.error.message
                for call in trace.calls
            )
            if first_prompt_too_long:
                # The harness builds the first prompt in the env server, so the dataset can't
                # be filtered for it; flag it here as a data problem, not a policy one.
                task = sample.verifiers_task_data.get(
                    "name", sample.verifiers_task_data.get("idx")
                )
                logger.warning(
                    "rollout group=%d/rollout=%d: the first prompt%s does not fit the "
                    "context; marking ERROR",
                    group_id,
                    rollout_id,
                    "" if task is None else f" of task {task}",
                )
            status = RolloutStatus.ERROR
        logs = verifiers_rollout_logs(verifiers_episode, trace)
        if status.is_error():
            log_failed_rollout(logs, group_id=group_id, rollout_id=rollout_id)
        else:
            turns[-1].env_rewards[VERIFIERS_REWARD_KEY] = trace.reward
        return Rollout(
            group_id=group_id,
            rollout_id=rollout_id,
            status=status,
            turns=turns,
            logs=logs,
        )

    @staticmethod
    def rollout_status(*, verifiers_episode: Any, trace: Any) -> RolloutStatus:
        # Verifiers flags an expired deadline with is_timeout: a stage's on its trace (which can keep
        # ok=True), the whole episode's on the episode.
        if trace.is_timeout or verifiers_episode.is_timeout:
            return RolloutStatus.ERROR_TIMEOUT
        if not verifiers_episode.ok or not trace.ok:
            # Verifiers records a failure as data instead of raising; the first error stopped the
            # rollout. Only a generator 4xx (e.g. a prompt past the context) or the agent's own
            # exit is the policy's doing. Any other error (a sandbox, env or tool failure, a
            # generator 5xx or failed connection) makes the rollout ERROR_SYSTEM.
            errors = trace.errors or verifiers_episode.errors
            if not errors:
                return RolloutStatus.ERROR
            error = errors[0]
            is_policy_error = error.type == VerifiersHarnessError.__name__ or (
                error.type == VerifiersProviderError.__name__
                and error.status_code is not None
                and error.status_code < 500
            )
            return (
                RolloutStatus.ERROR if is_policy_error else RolloutStatus.ERROR_SYSTEM
            )
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
        Non-sampled nodes after a sampled one, e.g. the tool result ``[5, 6]``,
        become that turn's ``env_messages``, also once per node.

        Verifiers does not return TorchTitan policy metadata, so every emitted
        turn receives the conservative min/max policy-version span accumulated
        by the generation server for the whole rollout. Generator metrics are
        attached once to avoid double counting. A turn's top-k rows come from the
        generation whose prompt length and completion tokens match its node.
        """
        if generation_metadata is None:
            if any(any(node.mask) for node in trace.nodes):
                raise ValueError(
                    "Verifiers trace has trainable tokens but no generation metadata"
                )
            return []

        node_index = {id(node): index for index, node in enumerate(trace.nodes)}
        trained_nodes: set[int] = set()
        replied_nodes: set[int] = set()
        last_turn_by_node: dict[int, RolloutTurn] = {}
        turns: list[RolloutTurn] = []
        # The last appended turn's prompt + completion; not reset between branches
        previous_token_ids: list[int] = []

        for branch in trace.branches:
            token_ids = branch.token_ids
            logprobs = branch.logprobs
            # [len(token_ids), num_layers, top_k] from the generation server; None without replay.
            # TODO: Verifiers gives a continued turn's last token the row of the token before it,
            # so 1 input per turn boundary replays its neighbor's experts (native: next prefill's row).
            routed_expert_ids = branch.routed_experts
            branch_offset = 0
            reply_to: RolloutTurn | None = None
            for node in branch.nodes:
                index = node_index[id(node)]
                mask = list(node.mask)
                if node.sampled and any(mask):
                    if index in trained_nodes:
                        mask = [False] * len(mask)
                    else:
                        trained_nodes.add(index)
                topk = None
                if generation_metadata.topk_by_generation and any(mask):
                    # The prompt this node's completion was sampled from ends at its first
                    # sampled token.
                    key = (
                        branch_offset + list(node.mask).index(True),
                        tuple(
                            token_id
                            for token_id, sampled in zip(node.token_ids, node.mask)
                            if sampled
                        ),
                    )
                    if key not in generation_metadata.topk_by_generation:
                        raise ValueError(
                            "Verifiers node has no generation with the same prompt length "
                            "and completion tokens, so its top-k logprobs are unknown"
                        )
                    topk = generation_metadata.topk_by_generation[key]
                for start, end in _trainable_token_spans(mask):
                    absolute_start = branch_offset + start
                    absolute_end = branch_offset + end
                    # Completion rows of this span: sampled tokens before `start` in the node.
                    row = sum(node.mask[:start])
                    prompt_prefix_len, prompt_delta_token_ids = split_prompt(
                        token_ids[:absolute_start], previous_token_ids
                    )
                    previous_token_ids = token_ids[:absolute_end]
                    turns.append(
                        RolloutTurn(
                            rollout_id=RolloutTurnID(
                                group_id=group_id,
                                rollout_id=rollout_id,
                                turn_id=len(turns),
                            ),
                            prompt_prefix_len=prompt_prefix_len,
                            prompt_delta_token_ids=prompt_delta_token_ids,
                            completion_token_ids=np.asarray(
                                token_ids[absolute_start:absolute_end], dtype=np.int32
                            ),
                            completion_logprobs=np.asarray(
                                logprobs[absolute_start:absolute_end], dtype=np.float32
                            ),
                            routed_expert_ids=(
                                None
                                if routed_expert_ids is None
                                else routed_expert_ids[
                                    max(prompt_prefix_len - 1, 0) : absolute_end - 1
                                ]
                            ),
                            completion_topk_token_ids=(
                                None
                                if topk is None
                                else topk[0][row : row + end - start]
                            ),
                            completion_topk_logprobs=(
                                None
                                if topk is None
                                else topk[1][row : row + end - start]
                            ),
                            min_policy_version=generation_metadata.min_policy_version,
                            max_policy_version=generation_metadata.max_policy_version,
                            completion_message=message_to_wire(node.message),
                            metrics=(
                                list(generation_metadata.metrics) if not turns else []
                            ),
                        )
                    )
                    last_turn_by_node[index] = turns[-1]
                if node.sampled:
                    reply_to = last_turn_by_node.get(index)
                elif reply_to is not None and index not in replied_nodes:
                    replied_nodes.add(index)
                    reply_to.env_messages.append(message_to_wire(node.message))
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


def verifiers_rollout_logs(verifiers_episode: Any, trace: Any) -> dict[str, Any]:
    """Summarize how a Verifiers rollout ended, for ``Rollout.logs``.

    Verifiers' own "rollout done" line prints only the error type or stop
    condition. This keeps the error messages and tracebacks (a failed
    harness's message ends with its stderr), per-phase wall time and model-call
    latency. For example, ``agent_sec=7200 model_sec=6900
    slowest_model_call_sec=1801`` on an agent timeout points at slow generation,
    not the task.
    """
    timing = trace.timing
    call_seconds = [call.time.duration for call in trace.calls]
    return {
        "verifiers_trace_id": trace.id,
        "task": trace.task.key,
        "stop_condition": trace.stop_condition,
        "errors": [
            error.model_dump(mode="json")
            for error in (*verifiers_episode.errors, *trace.errors)
        ],
        "setup_sec": timing.setup.duration,
        "agent_sec": timing.agent.duration,
        "model_sec": timing.agent.model.duration,
        "harness_sec": timing.agent.harness.duration,
        "scoring_sec": timing.scoring.duration,
        "model_calls": len(call_seconds),
        "failed_model_calls": sum(call.error is not None for call in trace.calls),
        "slowest_model_call_sec": max(call_seconds, default=0.0),
    }


def log_failed_rollout(logs: dict[str, Any], *, group_id: int, rollout_id: int) -> None:
    """Log the last error and timing from ``verifiers_rollout_logs`` in one line."""
    error = logs["errors"][-1] if logs["errors"] else {}
    traceback_tail = " | ".join(
        (error.get("traceback") or "").strip().splitlines()[-3:]
    )
    logger.warning(
        "Verifiers rollout failed: trace=%s group=%d rollout=%d task=%s stop=%s "
        "error=%s: %s | setup=%.0fs agent=%.0fs (model=%.0fs harness=%.0fs) "
        "scoring=%.0fs | model_calls=%d failed_calls=%d slowest_call=%.0fs | %s",
        logs["verifiers_trace_id"],
        group_id,
        rollout_id,
        logs["task"],
        logs["stop_condition"],
        error.get("type"),
        error.get("message"),
        logs["setup_sec"],
        logs["agent_sec"],
        logs["model_sec"],
        logs["harness_sec"],
        logs["scoring_sec"],
        logs["model_calls"],
        logs["failed_model_calls"],
        logs["slowest_model_call_sec"],
        traceback_tail,
    )
