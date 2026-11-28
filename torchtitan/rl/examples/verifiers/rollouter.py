# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rollouter backed by a Verifiers environment service."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace
from typing import Any, TYPE_CHECKING

import numpy as np
from verifiers.v1.configs.client import TrainClientConfig as VerifiersTrainClientConfig
from verifiers.v1.configs.taskset import TasksetConfig as VerifiersTasksetConfig
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.serve.client import EnvClient as VerifiersEnvClient
from verifiers.v1.types import SamplingConfig as VerifiersSamplingConfig

from torchtitan.components.data.multimodal.vision_processor import VisionProcessor
from torchtitan.components.renderer import RendererConfig, RenderersConfigAdapter
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.rl.examples.verifiers.data import (
    VerifiersTaskDataset,
    VerifiersTaskSample,
)
from torchtitan.rl.examples.verifiers.env_server import VerifiersEnvServer
from torchtitan.rl.examples.verifiers.generation_server import (
    GenerationServer,
    GROUP_ID_SAMPLING_PARAM,
    VerifiersGenerationMetadata,
)
from torchtitan.rl.rollout.data import IterableRolloutDataLoader
from torchtitan.rl.rollout.rollouter import (
    BaseRolloutWorker,
    log_failed_rollout,
    Rollouter,
)
from torchtitan.rl.rollout.rubric import RewardFn
from torchtitan.rl.rollout.types import (
    error_entry,
    GenerateFn,
    Prompt,
    ReleaseSessionFn,
    Rollout,
    RolloutGroup,
    RolloutStatus,
    RolloutTurn,
    RolloutTurnID,
    split_prompt,
)
from torchtitan.tools.garbage_collector import GarbageCollector

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


class VerifiersRollouter(Rollouter):
    """Runs Verifiers envs as TitanRL rollout groups.

    TitanRL provides: the tasks (`VerifiersTaskDataset`), the model (`generate_fn`, which each
    worker's `GenerationServer` serves to Verifiers) and the training (rewards, advantages).
    Verifiers provides: the env, the agent loop, the sandbox and the grade.

    This rollouter starts the Verifiers env server, then the rollout workers
    (`VerifiersRolloutWorker`) that send it each group's task. The diagram: "Integration flow" in
    this folder's README.md.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Rollouter.Config):
        training_dataloader: IterableRolloutDataLoader.Config
        """Verifiers taskset used for training samples and environment scoring."""

        validation_dataset: VerifiersTaskDataset.Config
        """Verifiers taskset used for validation samples."""

        worker: VerifiersRolloutWorker.Config
        """How a rollout worker runs a Verifiers group; `setup_async` passes each worker the
        EnvServer's address."""

        verifiers_env_server: VerifiersEnvServer.Config
        """Environment server spawned and owned by this rollouter."""

        def __post_init__(self) -> None:
            Rollouter.Config.__post_init__(self)
            # The user sets the taskset once, in the training dataset; the env server's comes from it.
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

    def __init__(self, config: Config, *, max_rollouts_in_flight: int) -> None:
        super().__init__(config, max_rollouts_in_flight=max_rollouts_in_flight)
        self._verifiers_env_server = config.verifiers_env_server.build()

    async def setup_async(
        self,
        *,
        actor_system: ActorSystem,
        tokenizer_config: HuggingFaceTokenizer.Config,
        renderer_config: RendererConfig,
        hf_assets_path: str,
        vision_processor_config: VisionProcessor.Config | None = None,
    ) -> None:
        """Start the EnvServer, then the rollout workers that connect to it."""
        if vision_processor_config is not None:
            # TODO: pass the vision processor to the renderer the EnvServer builds.
            raise NotImplementedError(
                "Verifiers environments do not support image prompts yet; unset "
                "Controller.Config.vision_processor."
            )
        # Checked here too, so a wrong renderer fails before the env server or a rollout worker starts.
        if not isinstance(renderer_config, RenderersConfigAdapter):
            raise ValueError("Verifiers requires a renderer from from_renderers(...)")
        env_server_address = await self._verifiers_env_server.start()
        await super().setup_async(
            actor_system=actor_system,
            tokenizer_config=tokenizer_config,
            renderer_config=renderer_config,
            hf_assets_path=hf_assets_path,
            worker_kwargs={"env_server_address": env_server_address},
        )

    async def close(self) -> None:
        """Stop the rollout workers, then the EnvServer.

        Workers first: `Rollouter.close` waits for each running group, and a group waits for its
        episodes' results from the EnvServer. A stopped EnvServer never sends them, so each group
        would wait out `_RUN_GROUP_DRAIN_TIMEOUT_S`.
        """
        try:
            await super().close()
        finally:
            await self._verifiers_env_server.close()


class VerifiersRolloutWorker(BaseRolloutWorker):
    """Runs Verifiers groups in a rollout worker process: the env server runs each rollout, and
    this worker serves its model calls and turns its trace into a TitanRL rollout."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseRolloutWorker.Config):
        generation_server: GenerationServer.Config
        """Local HTTP bridge from Verifiers to TitanRL generation; each rollout worker runs one."""

        renderer_multiplex: int = 256
        """Maximum concurrent rollouts sharing one Verifiers renderer instance.

        TODO: evaluate sharing this renderer pool with TitanRL's native rollout
        path instead of keeping it specific to the Verifiers client.
        """

        connection_timeout_sec: float = 120.0
        """Maximum time to wait for the Verifiers server to become healthy."""

        def __post_init__(self) -> None:
            if self.renderer_multiplex <= 0:
                raise ValueError("renderer_multiplex must be positive")
            if self.connection_timeout_sec <= 0:
                raise ValueError("connection_timeout_sec must be positive")

    def __init__(self, config: Config, *, env_server_address: str) -> None:
        super().__init__(config)
        self._config = config
        # The EnvServer's ZMQ address, passed by `VerifiersRollouter.setup_async`.
        # TODO: remove it with the env server, once the Verifiers env runs in the rollout worker.
        self._env_server_address = env_server_address
        self._generation_server: GenerationServer = config.generation_server.build()
        self._verifiers_env_client: VerifiersEnvClient
        self._verifiers_train_client_config: VerifiersTrainClientConfig

    async def setup_async(
        self,
        *,
        tokenizer_config: HuggingFaceTokenizer.Config,
        renderer_config: RendererConfig,
        hf_assets_path: str,
        vision_processor_config: VisionProcessor.Config | None = None,
    ) -> None:
        """Start this worker's generation server and connect to the EnvServer. Verifiers
        renders in the env server, so no tokenizer here. `VerifiersRollouter.setup_async`
        rejects a vision processor."""
        del tokenizer_config, vision_processor_config
        if not isinstance(renderer_config, RenderersConfigAdapter):
            raise ValueError(
                "Verifiers requires a renderer configured with from_renderers(...) "
                "so its client can construct the same renderer in the "
                "environment-server process"
            )
        # The EnvServer and the rollout workers both run on the controller host, so the
        # agents reach this server on the loopback address.
        await self._generation_server.start()
        self._verifiers_env_client = VerifiersEnvClient(self._env_server_address)
        await self._verifiers_env_client.wait_for_server_startup(
            timeout=self._config.connection_timeout_sec
        )
        self._verifiers_train_client_config = VerifiersTrainClientConfig(
            base_url=self._generation_server.base_url,
            # The local endpoint does not authenticate. An empty environment
            # variable name makes Verifiers send its required "EMPTY" value.
            api_key_var="",
            renderer=renderer_config.get_renderers_config(),
            multiplex=self._config.renderer_multiplex,
            renderer_model_name=hf_assets_path,
        )

        # The end of `BaseRolloutWorker.setup_async`, which this worker skips: Verifiers builds its own renderer.
        GarbageCollector.freeze(
            "Full GC before freezing the rollout worker's setup objects"
        )
        self.start_process_metrics()

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
        # Each group replaces this one slot. That is safe because every group's generate_fn routes the
        # same way; keep per-group state out of it.
        self._generation_server.set_generate_fn(generate_fn)

        # Run the group's rollouts in the env server, all at once.
        async with asyncio.TaskGroup() as task_group:
            tasks = [
                task_group.create_task(
                    self._run_single_rollout(
                        release_session_fn=release_session_fn,
                        sample=sample,
                        sampling=(
                            sampling
                            if sampling.seed is None
                            else replace(sampling, seed=sampling.seed + rollout_id)
                        ),
                        group_id=group_id,
                        rollout_id=rollout_id,
                    )
                )
                for rollout_id in range(group_size)
            ]
        group = await self.make_scored_group(
            rollouts=[task.result() for task in tasks], sample=sample, group_id=group_id
        )
        group.metrics += self.flush_process_metrics(group_id)
        return group

    async def _run_single_rollout(
        self,
        *,
        release_session_fn: ReleaseSessionFn,
        sample: object,
        sampling: SamplingConfig,
        group_id: int,
        rollout_id: int,
    ) -> Rollout:
        """Send one task to Verifiers, release its sessions, and convert its trace to a rollout.
        An exception ends it as ERROR_INFRA. A failed rollout is logged once, at the end.
        """

        if not isinstance(sample, VerifiersTaskSample):
            raise TypeError("Verifiers requires a VerifiersTaskSample")

        try:
            # ======== Run the whole rollout in Verifiers ========
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
                    **{GROUP_ID_SAMPLING_PARAM: group_id},
                ),
            )

            # ======== Release each trace's routing session ========
            # For each trace, release its routing session (Verifiers sends the trace id as X-Session-ID)

            # TODO: an attempt Verifiers retried has no trace here, so its held KV waits for the
            # group's release. The generation server could record each rollout's session ids and
            # release those too.
            generation_metadata_by_session = {
                trace.id: self._generation_server.pop_generation_metadata(trace.id)
                for trace in verifiers_episode.traces
            }

            async def release(session_id: str) -> None:
                try:
                    await release_session_fn(routing_session_id=session_id)
                except Exception:
                    # Not fatal: `max_sessions` evicts an unreleased session.
                    logger.exception("releasing routing session %s failed", session_id)

            await asyncio.gather(
                *(release(trace.id) for trace in verifiers_episode.traces)
            )

            # ======== Convert the trainable trace to turns ========
            traces = [
                trace for trace in verifiers_episode.traces if trace.agent.trainable
            ]
            # TODO: allow more than one trainable trace per episode, which happens in e.g. Verifiers'
            #   `best_of_n` env (n attempts per episode).
            if len(traces) != 1:
                raise ValueError(
                    "Verifiers expects one trainable trace per rollout; got "
                    f"{len(traces)}"
                )

            # Convert Verifiers' graph trace to TitanRL's linear rollout turns.
            trace = traces[0]
            generation_metadata = generation_metadata_by_session[trace.id]
            turns = self.trace_to_rollout_turns(
                trace=trace,
                generation_metadata=generation_metadata,
                group_id=group_id,
                rollout_id=rollout_id,
            )

            # ======== Status, reward and logs ========
            status = self.rollout_status(
                verifiers_episode=verifiers_episode, trace=trace
            )

            # TODO: Many of these should be metrics, not logs.
            logs = verifiers_rollout_logs(verifiers_episode, trace)

            # No turn: nothing the policy wrote can train (e.g. no model call has a trainable
            # token), so don't train it. A status that already doesn't train is kept.
            if not turns and status.is_trainable():
                status = RolloutStatus.ERROR_INFRA
                logs["errors"].append(
                    {
                        "type": "no_trainable_token",
                        "message": f"{trace.id}: no model call has a trainable token",
                    }
                )

            if turns and not status.is_error():
                # Rewards by the env, if any, are added to the last turn.
                # The rollout will be scored later.
                # TODO: provide better support to carry rewards per trace
                turns[-1].env_rewards[VERIFIERS_REWARD_KEY] = trace.reward

            rollout = Rollout(
                group_id=group_id,
                rollout_id=rollout_id,
                status=status,
                turns=turns,
                logs=logs,
            )
        except Exception as error:
            # e.g. the env server is unreachable, or the trace can't be converted
            rollout = Rollout(
                group_id=group_id,
                rollout_id=rollout_id,
                status=RolloutStatus.ERROR_INFRA,
                logs={"errors": [error_entry(error)]},
            )

        log_failed_rollout(rollout)

        return rollout

    @staticmethod
    def rollout_status(*, verifiers_episode: Any, trace: Any) -> RolloutStatus:
        """Map a Verifiers trace to its status. `RolloutStatus` says which statuses train.

        status                     the trace ...
        COMPLETED                  finished, and so did its episode
        ERROR                      failed on a HarnessError or TaskError, e.g. the agent's `pkill -f`
                                   killed its harness
        ERROR_INFRA                ended before the policy's first answer, failed on any other error
                                   (e.g. SandboxError, a 502), or finished in an episode that failed
                                   (e.g. a judge)
        ERROR_TIMEOUT              missed a deadline after the policy's first answer (agent, finalize, scoring)
        ERROR_UNKNOWN              was truncated for a reason not mapped here, e.g. Verifiers' `compaction_failed`
        FIRST_PROMPT_TOO_LONG      got a 400 on its first model call (the prompt reached `max_rollout_tokens`)
        TRUNCATED_LENGTH           ended after a completion hit `max_tokens`, or used up a Verifiers token
                                   budget (`max_input_tokens`, `max_output_tokens`, `max_total_tokens`)
        TRUNCATED_MAX_TURNS        hit max_turns
        TRUNCATED_PROMPT_TOO_LONG  ended on a 400 on a later model call (the prompt reached `max_rollout_tokens`)
        """
        # TODO: an error type or a deadline says where a rollout broke, not who broke it.
        # We could add more granularity e.g. Harbor wraps a sandbox error in a TaskError.
        last_call_error = trace.calls[-1].error if trace.calls else None
        refused = last_call_error is not None and last_call_error.status_code == 400

        # A first prompt that doesn't fit can't train, so nothing after it changes the status.
        if refused and trace.num_turns == 0:
            return RolloutStatus.FIRST_PROMPT_TOO_LONG

        # The policy never answered (e.g. the sandbox didn't start, or the harness died before
        # its first model call), so the failure isn't the policy's, and there is nothing to train.
        if trace.num_turns == 0:
            return RolloutStatus.ERROR_INFRA

        if trace.is_timeout or verifiers_episode.is_timeout:
            return RolloutStatus.ERROR_TIMEOUT

        if trace.ok and not verifiers_episode.ok:
            return RolloutStatus.ERROR_INFRA

        if refused and (trace.ok or trace.last_error.status_code == 400):
            return RolloutStatus.TRUNCATED_PROMPT_TOO_LONG

        if not trace.ok:
            # A failed trace always records its error. The newest one ended the trace.
            if trace.last_error.type in ("HarnessError", "TaskError"):
                return RolloutStatus.ERROR
            return RolloutStatus.ERROR_INFRA

        if not trace.is_truncated:
            return RolloutStatus.COMPLETED

        if trace.stop_condition == "max_turns":
            return RolloutStatus.TRUNCATED_MAX_TURNS

        # TODO: support more granularity
        if trace.stop_condition in (
            "max_input_tokens",
            "max_output_tokens",
            "max_total_tokens",
        ):
            return RolloutStatus.TRUNCATED_LENGTH

        if trace.calls and trace.calls[-1].finish_reason == "length":
            return RolloutStatus.TRUNCATED_LENGTH

        return RolloutStatus.ERROR_UNKNOWN

    @staticmethod
    def trace_to_rollout_turns(
        *,
        trace: Any,
        generation_metadata: VerifiersGenerationMetadata | None,
        group_id: int,
        rollout_id: int,
    ) -> list[RolloutTurn]:
        """Flatten a Verifiers trace into TitanRL rollout turns, one per span of trainable tokens.

        Example, one branch: user [1, 2], sampled [3, 4], tool result [5, 6], sampled [7, 8].

            turn 0   prompt [1, 2]               completion [3, 4]   env_messages [tool result]
            turn 1   prompt [1, 2, 3, 4, 5, 6]   completion [7, 8]

        A node shared by several branches is emitted once. Every turn gets the rollout-wide
        min/max policy version from the generation server; generator metrics go on turn 0 only.
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
                for start, end in _trainable_token_spans(mask):
                    absolute_start = branch_offset + start
                    absolute_end = branch_offset + end
                    prompt_prefix_len, prompt_delta_token_ids = split_prompt(
                        Prompt(token_ids=token_ids[:absolute_start]),
                        previous=Prompt(token_ids=previous_token_ids),
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
        # Retried attempts' errors too, so one rollout can add several to `rollout/errors/<type>`.
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
