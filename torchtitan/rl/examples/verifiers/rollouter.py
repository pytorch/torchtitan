# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A TitanRL rollout worker that runs Verifiers envs."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, TYPE_CHECKING

import numpy as np
import verifiers.v1 as vf
from verifiers.v1.configs.taskset import TasksetConfig as VerifiersTasksetConfig
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.env import RunSlot

from torchtitan.components.data.multimodal.vision_processor import VisionProcessor
from torchtitan.components.renderer import RendererConfig, RenderersConfigAdapter
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.rl.examples.verifiers.data import VerifiersTaskSample
from torchtitan.rl.examples.verifiers.generation_server import (
    GenerationServer,
    VerifiersGenerationMetadata,
)
from torchtitan.rl.examples.verifiers.worker_setup import register_local_taskset_alias
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout.rollouter import BaseRolloutWorker, log_failed_rollout
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

if TYPE_CHECKING:
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


def _default_run_id() -> str:
    """The launcher's `VF_RUN_ID`, else a new one. Built once with the config, so all the
    workers of a run share it."""
    return os.environ.get("VF_RUN_ID") or f"torchtitan-{uuid.uuid4().hex}"


class VerifiersRolloutWorker(BaseRolloutWorker):
    """Runs Verifiers envs in a TitanRL rollout worker.

        TitanRL provides     the tasks (`VerifiersTaskDataset`), the model, rewards, advantages
        Verifiers provides   the env, the agent loop, the sandbox, the grade

    The model reaches Verifiers through this worker's `GenerationServer`, which calls `generate_fn`.

    One group = one task, run `group_size` times with `env.run_slot`. Each episode's trainable
    trace becomes one TitanRL `Rollout`. How the pieces connect: "Integration flow" in this
    folder's README.md.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseRolloutWorker.Config):
        # TODO: enable multi-env training: load the env from each sample's config, so one job can mix
        # tasksets. Needs the dataset to carry each sample's env config. Until then, nothing checks that
        # this taskset matches the training or validation dataset's, though the worker builds both
        # datasets' tasks with it.
        environment: vf.EnvConfig
        """The Verifiers env each worker loads: its taskset, agent, harness and runtime. Give it
        the same taskset object as the training dataset. The validation dataset's taskset must
        have the same task class and task config: its tasks are built with this one."""

        generation_server: GenerationServer.Config
        """Local HTTP bridge from Verifiers to TitanRL generation; each rollout worker runs one."""

        run_id: str = field(default_factory=_default_run_id)
        """Verifiers' run id (`VF_RUN_ID`). Default: the launcher's `VF_RUN_ID`, else a new id.
        It labels the run's sandboxes, e.g. Docker containers get `verifiers.run=<run_id>`, and
        scopes Verifiers' sandbox-creation limiter, e.g. for Prime sandboxes. The limiter keeps
        its state under `~/.cache/verifiers`, so workers on other hosts share it only if that
        folder is shared."""

        renderer_multiplex: int = 256
        """Maximum concurrent rollouts sharing one Verifiers renderer instance.

        TODO: evaluate sharing this renderer pool with TitanRL's native rollout
        path instead of keeping it specific to the Verifiers client.
        """

        def __post_init__(self) -> None:
            if self.renderer_multiplex <= 0:
                raise ValueError("renderer_multiplex must be positive")

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._config = config
        self._generation_server: GenerationServer = config.generation_server.build()
        # A taskset under a local alias id can be imported only after this process registers it.
        taskset_module = _local_taskset_module(config.environment.taskset)
        if taskset_module is not None:
            register_local_taskset_alias(taskset_module)
        # Verifiers' `run_scope()` reads `VF_RUN_ID` from the process env on each call, so
        # setting it after import works. Every worker sets the config's id before loading its
        # env, so one label finds the run's sandboxes across workers.
        os.environ["VF_RUN_ID"] = config.run_id
        self._verifiers_env = vf.load_environment(
            vf.resolve_env_config(config.environment)
        )
        # Holds `env.serving()` open from `setup_async` to `close`. Those are two calls, so a `with`
        # block can't span them. Step 3 of `setup_async` says what it serves.
        self._serving = contextlib.AsyncExitStack()
        # Rollouts still running, so `close` can cancel them.
        self._running_rollouts: set[asyncio.Task[Rollout]] = set()
        self._train_client_config: vf.TrainClientConfig

    async def setup_async(
        self,
        *,
        tokenizer_config: HuggingFaceTokenizer.Config,
        renderer_config: RendererConfig,
        hf_assets_path: str,
        vision_processor_config: VisionProcessor.Config | None = None,
    ) -> None:
        """Start the local servers this worker's episodes use:

        1. `no_proxy` lists 127.0.0.1, so the local calls below skip the host's HTTP proxy.
        2. Our generation server: Verifiers' model calls -> `generate_fn`.
        3. `env.serving()`: Verifiers' interception server (the agent's model calls go through it)
           and its shared tools. It must stay open until `close`, so an `AsyncExitStack` holds it
           across the two calls.

        Verifiers builds its renderer from `renderer_config` and the tokenizer files in
        `hf_assets_path`, so `tokenizer_config` is unused.
        """
        del tokenizer_config
        if vision_processor_config is not None:
            # TODO: pass the vision processor to the renderer this worker's train client builds.
            raise NotImplementedError(
                "Verifiers environments do not support image prompts yet; unset "
                "Controller.Config.vision_processor."
            )
        if not isinstance(renderer_config, RenderersConfigAdapter):
            raise ValueError(
                "Verifiers' train client builds the renderer from a `renderers` config, the same "
                "config TitanRL uses: configure the renderer with from_renderers(...)"
            )
        # Verifiers' HTTP client sends even a 127.0.0.1 call through `http_proxy`, unless
        # `no_proxy` names 127.0.0.1. Add the loopback names to both spellings, keeping what
        # either already holds, e.g. ".corp.internal" -> ".corp.internal,127.0.0.1,localhost".
        no_proxy_entries = (
            os.environ.get("no_proxy"),
            os.environ.get("NO_PROXY"),
            "127.0.0.1",
            "localhost",
        )
        no_proxy = ",".join(dict.fromkeys(entry for entry in no_proxy_entries if entry))
        os.environ["no_proxy"] = os.environ["NO_PROXY"] = no_proxy
        # Verifiers' train client runs in this process, so it calls this server on the loopback
        # address.
        await self._generation_server.start()
        await self._serving.enter_async_context(self._verifiers_env.serving())
        self._train_client_config = vf.TrainClientConfig(
            base_url=self._generation_server.base_url,
            # The local endpoint does not authenticate. An empty environment
            # variable name makes Verifiers send its required "EMPTY" value.
            api_key_var="",
            renderer=renderer_config.get_renderers_config(),
            multiplex=self._config.renderer_multiplex,
            renderer_model_name=hf_assets_path,
        )

    async def close(self) -> None:
        """Cancel the rollouts still running, then stop serving the env and the generation server.
        Each cancelled `run_slot` stops its own sandbox, e.g. removes its Docker container."""
        # TODO: a killed worker never runs `close`, so its running sandboxes (e.g. Docker
        # containers) stay up. Removing them needs a cleanup that runs outside the worker.
        running_rollouts = list(self._running_rollouts)
        for rollout_task in running_rollouts:
            rollout_task.cancel()
        await asyncio.gather(*running_rollouts, return_exceptions=True)
        await self._serving.aclose()
        await self._generation_server.close()

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
        """Run one task `group_size` times through Verifiers, then score the group."""
        # The generation server holds one `generate_fn`, and each group sets it again. Safe: every
        # group's `generate_fn` routes the same way. Keep per-group state out of it.
        self._generation_server.set_generate_fn(generate_fn)

        # ======== Build the task ========
        if not isinstance(sample, VerifiersTaskSample):
            raise TypeError(
                f"Verifiers rollouts need a VerifiersTaskSample, got {type(sample).__name__}"
            )
        # The sample carries only the task's data; the env's taskset gives its class and config.
        task_type = self._verifiers_env.taskset.task_type()
        task = task_type(
            data=task_type.data_type().model_validate(sample.verifiers_task_data),
            config=self._verifiers_env.config.taskset.task,
        )

        # ======== Run the episodes ========
        # Verifiers plans one `RunSlot` per episode; `run_slot` runs it with the env's retries.
        slots = self._verifiers_env.slots(task, n=group_size)
        async with asyncio.TaskGroup() as task_group:
            rollout_tasks = [
                task_group.create_task(
                    self._run_single_rollout(
                        release_session_fn=release_session_fn,
                        slot=slot,
                        sampling=(
                            sampling
                            if sampling.seed is None
                            else replace(sampling, seed=sampling.seed + rollout_id)
                        ),
                        group_id=group_id,
                        rollout_id=rollout_id,
                    )
                )
                for rollout_id, slot in enumerate(slots)
            ]
            for rollout_task in rollout_tasks:
                self._running_rollouts.add(rollout_task)
                rollout_task.add_done_callback(self._running_rollouts.discard)

        # ======== Score the group ========
        return await self.make_scored_group(
            rollouts=[rollout_task.result() for rollout_task in rollout_tasks],
            sample=sample,
            group_id=group_id,
        )

    async def _run_single_rollout(
        self,
        *,
        release_session_fn: ReleaseSessionFn,
        slot: RunSlot,
        sampling: SamplingConfig,
        group_id: int,
        rollout_id: int,
    ) -> Rollout:
        """Run one Verifiers episode, release its sessions, and convert its trace to a rollout.
        An exception ends it as ERROR_INFRA. A failed rollout is logged once, at the end.
        """

        try:
            # ======== Run the episode and release its sessions ========
            model_context = vf.ModelContext(
                client=self._train_client_config,
                model=self._generation_server.model_id,
                sampling=vf.SamplingConfig(
                    temperature=sampling.temperature,
                    top_p=sampling.top_p,
                    max_tokens=sampling.max_tokens,
                    seed=sampling.seed,
                ),
            )

            # Our generation server finds each model call's group by its session id: the
            # X-Session-ID header, which is the trace id. A retried attempt creates another trace,
            # e.g. session_ids == ["9f3c", "a71b"].
            session_ids: list[str] = []

            def start_session(trace: vf.Trace) -> None:
                # Verifiers calls this before the trace's first model call.
                session_ids.append(trace.id)
                self._generation_server.start_session(
                    session_id=trace.id, group_id=group_id
                )

            try:
                verifiers_episode = await self._verifiers_env.run_slot(
                    slot=slot, ctx=model_context, on_trace=start_session
                )
            finally:
                # TODO: release a session only after its model calls finish. Today this can happen:
                #   1. Verifiers gives up on a model call (e.g. at its deadline), but our generation
                #      server is still running it.
                #   2. The release below tells the router the session is done.
                #   3. The abandoned call then reaches the router, which sends the session to a
                #      generator again.
                #   4. That generator keeps the session's KV cache until the group's release.
                #   Fix: the generation server counts each session's running calls, and the release
                #   waits for the last one.
                generation_metadata_by_session = {
                    session_id: self._generation_server.end_session(session_id)
                    for session_id in session_ids
                }
                await asyncio.gather(
                    *(
                        release_session_fn(routing_session_id=session_id)
                        for session_id in session_ids
                    )
                )

            # ======== Pick the policy's trace ========
            # One trace per agent, e.g. the policy and a simulated user. Exactly one is trainable,
            # the policy's: keep it. A retried attempt's traces are gone; Verifiers keeps only the
            # last attempt.
            trainable_traces = [
                trace for trace in verifiers_episode.traces if trace.agent.trainable
            ]
            # TODO: allow more than one trainable trace per episode, which happens in e.g. Verifiers'
            #   `best_of_n` env (n attempts per episode).
            if len(trainable_traces) != 1:
                raise ValueError(
                    "Verifiers expects one trainable trace per rollout; got "
                    f"{len(trainable_traces)}"
                )
            (trace,) = trainable_traces

            # ======== Convert it to turns and a status ========
            # Verifiers' graph trace becomes TitanRL's linear rollout turns.
            turns = self.trace_to_rollout_turns(
                trace=trace,
                generation_metadata=generation_metadata_by_session[trace.id],
                group_id=group_id,
                rollout_id=rollout_id,
            )
            status = self.rollout_status(
                verifiers_episode=verifiers_episode, trace=trace
            )

            # TODO: Many of these should be metrics, not logs.
            logs = verifiers_rollout_logs(
                verifiers_episode=verifiers_episode, trace=trace
            )

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

            # ======== Record the reward and logs ========
            # Log Verifiers' named rewards and metrics on turn 0, e.g. verifiers/reward/solved.
            # - Every rollout with turns logs them, whatever its status.
            # - A reward is its raw score; the training reward, `trace.reward`, is their weighted sum.
            # - None means Verifiers never computed it (e.g. scoring didn't run): skip it.
            if turns:
                turns[0].metrics.extend(
                    m.Metric(f"verifiers/reward/{name}", m.Mean(reward.score))
                    for name, reward in trace.rewards.items()
                    if reward is not None
                )
                turns[0].metrics.extend(
                    m.Metric(f"verifiers/metric/{name}", m.Mean(value))
                    for name, value in trace.metrics.items()
                    if value is not None
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
            # e.g. `run_slot` raised, or the trace can't be converted
            rollout = Rollout(
                group_id=group_id,
                rollout_id=rollout_id,
                status=RolloutStatus.ERROR_INFRA,
                logs={"errors": [error_entry(error)]},
            )

        log_failed_rollout(rollout)

        return rollout

    @staticmethod
    def rollout_status(
        *, verifiers_episode: vf.Episode, trace: vf.Trace
    ) -> RolloutStatus:
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
        trace: vf.Trace,
        generation_metadata: VerifiersGenerationMetadata | None,
        group_id: int,
        rollout_id: int,
    ) -> list[RolloutTurn]:
        """Flatten a Verifiers trace into TitanRL rollout turns, one per span of trainable tokens.

        Example, one branch: user [1, 2], sampled [3, 4], tool result [5, 6], sampled [7, 8].
        A turn stores its prompt as the length it shares with the previous turn's prompt +
        completion (`prompt_prefix_len`), plus the new tokens (`prompt_delta_token_ids`):

            turn 0   prefix 0, delta [1, 2]   completion [3, 4]   env_messages [tool result]
            turn 1   prefix 4, delta [5, 6]   completion [7, 8]

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


def verifiers_rollout_logs(
    verifiers_episode: vf.Episode, trace: vf.Trace
) -> dict[str, Any]:
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
