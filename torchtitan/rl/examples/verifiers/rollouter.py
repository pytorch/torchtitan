# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A TitanRL rollout worker that runs Verifiers envs.

How the pieces connect, step by step: README.md in this folder.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import uuid
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np
import verifiers.v1 as vf
from verifiers.v1.dialects.chat import message_to_wire

from torchtitan.components.data.multimodal.vision_processor import VisionProcessor
from torchtitan.components.renderer import RendererConfig, RenderersConfigAdapter
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.observability import structured_logger as sl
from torchtitan.rl.examples.verifiers.data import VerifiersTaskSample
from torchtitan.rl.examples.verifiers.generation_server import (
    GenerationServer,
    GenerationSession,
)
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout.rollouter import BaseRolloutWorker
from torchtitan.rl.rollout.rubric import RewardFn, Rubric, RubricOutput
from torchtitan.rl.rollout.types import (
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

# How long `close` waits for cancelled rollouts to stop their sandboxes. Docker removes a container
# in seconds; a teardown stuck longer must not hold up the job's exit.
_CLOSE_TIMEOUT_S = 30.0


class RewardFromVerifiers(RewardFn):
    """The grade Verifiers gave the rollout (`Trace.reward`), kept on its last turn.
    0.0 for a rollout with no turns."""

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        pass

    async def __call__(self, rollout: Rollout, env_input: object) -> float:
        del env_input
        return (
            rollout.turns[-1].env_rewards[VERIFIERS_REWARD_KEY]
            if rollout.turns
            else 0.0
        )


def _default_run_id() -> str:
    """The launcher's `VF_RUN_ID`, else a new one. Built once with the config, so all the
    workers of a run share it."""
    return os.environ.get("VF_RUN_ID") or f"torchtitan-{uuid.uuid4().hex}"


class VerifiersRolloutWorker(BaseRolloutWorker):
    """Runs Verifiers envs in a TitanRL rollout worker.

    TitanRL provides: the tasks (`VerifiersTaskDataset`), the model (`generate_fn`, which this
    worker's `GenerationServer` serves to Verifiers) and the training (rewards, advantages).
    Verifiers provides: the env, the agent loop, the sandbox and the grade.

    One group = one task, run `group_size` times with `env.run_slot`. Each episode's trainable
    trace becomes one TitanRL `Rollout`.

    Example:
        VerifiersRolloutWorker.Config(max_rollout_tokens=32768)
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseRolloutWorker.Config):
        max_rollout_tokens: int
        """The generator's context length. A model call whose prompt leaves no room for one
        output token is refused before it reaches the generator."""

        run_id: str = field(default_factory=_default_run_id)
        """Verifiers' run id (`VF_RUN_ID`). It labels the run's sandboxes, e.g. its Docker
        containers, and scopes Verifiers' run-wide limits, e.g. `creates_per_min`."""

        rubric: Rubric.Config = field(
            default_factory=lambda: Rubric.Config(
                reward_fns=[RewardFromVerifiers.Config()]
            )
        )
        """Verifiers' grade as each trained rollout's reward. Which rollouts train: the "Failed
        rollouts" table in this folder's README."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._config = config
        self._generation_server = GenerationServer(
            max_rollout_tokens=config.max_rollout_tokens
        )
        self._release_session_fn: ReleaseSessionFn
        self._client_config: vf.TrainClientConfig
        # One env per env config, built on its first group. `close` stops serving them.
        self._envs: dict[str, vf.Env] = {}
        self._envs_lock = asyncio.Lock()
        self._serving_envs = contextlib.AsyncExitStack()
        # Episodes still running, so `close` can cancel them.
        self._episode_tasks: set[asyncio.Task] = set()

    async def setup_async(
        self,
        *,
        generate_fn: GenerateFn,
        release_session_fn: ReleaseSessionFn,
        tokenizer_config: HuggingFaceTokenizer.Config,
        renderer_config: RendererConfig,
        hf_assets_path: str,
        vision_processor_config: VisionProcessor.Config | None = None,
    ) -> None:
        """Start the generation server. Verifiers renders prompts with its own copy of the
        renderer, so `tokenizer_config` is unused."""
        del tokenizer_config
        if vision_processor_config is not None:
            # TODO: pass the vision processor to the renderer Verifiers builds in this worker.
            raise NotImplementedError(
                "Verifiers rollouts don't support image prompts yet; unset "
                "Controller.Config.vision_processor"
            )
        if not isinstance(renderer_config, RenderersConfigAdapter):
            raise ValueError(
                "Verifiers builds its own copy of the renderer: configure it with "
                "from_renderers(...)"
            )
        os.environ["VF_RUN_ID"] = self._config.run_id
        _bypass_proxy_for_loopback()
        self._release_session_fn = release_session_fn
        base_url = await self._generation_server.start(generate_fn)
        self._client_config = vf.TrainClientConfig(
            base_url=base_url,
            # The loopback server checks no key; an empty variable name sends Verifiers' "EMPTY".
            api_key_var="",
            renderer=renderer_config.get_renderers_config(),
            renderer_model_name=hf_assets_path,
        )

    async def close(self) -> None:
        """Stop the worker, in this order:

        1. Cancel the episodes still running, and wait up to `_CLOSE_TIMEOUT_S` for them. Each
           cancelled `run_slot` stops its own sandbox.
        2. Stop serving the envs.
        3. Stop the generation server, even if step 2 raised.

        Each step stops only what is still up, so a close before or during setup, or during
        another close, returns at once.
        """
        episode_tasks, self._episode_tasks = self._episode_tasks, set()
        for task in episode_tasks:
            task.cancel()
        if episode_tasks:
            _, still_running = await asyncio.wait(
                episode_tasks, timeout=_CLOSE_TIMEOUT_S
            )
            if still_running:
                logger.warning(
                    "%d episodes did not stop within %.0f s; their sandboxes may outlive "
                    "this worker",
                    len(still_running),
                    _CLOSE_TIMEOUT_S,
                )
        try:
            await self._serving_envs.aclose()
        finally:
            await self._generation_server.close()

    async def run_group(
        self,
        *,
        sample: object,
        group_id: int,
        group_size: int,
        sampling: SamplingConfig,
    ) -> RolloutGroup:
        """Run one task `group_size` times through Verifiers, then score the group."""
        if not isinstance(sample, VerifiersTaskSample):
            raise TypeError(
                f"Verifiers rollouts need a VerifiersTaskSample, got {type(sample).__name__}"
            )
        env = await self._serving_env(sample.env_config_json)
        # The taskset's task class rebuilds the task from its data, as its `load` built it.
        task_type = env.taskset.task_type()
        task = task_type(
            data=task_type.data_type().model_validate(sample.task_data),
            config=env.taskset.config.task,
        )
        episode_tasks = [
            asyncio.create_task(
                self._run_episode(
                    env=env,
                    task=task,
                    # Each sibling gets its own seed, so the group is diverse yet reproducible.
                    sampling=(
                        sampling
                        if sampling.seed is None
                        else replace(sampling, seed=sampling.seed + episode_index)
                    ),
                    group_id=group_id,
                    episode_index=episode_index,
                )
            )
            for episode_index in range(group_size)
        ]
        for episode_task in episode_tasks:
            self._episode_tasks.add(episode_task)
            episode_task.add_done_callback(self._episode_tasks.discard)
        # return_exceptions: every sibling ends before the controller frees the group's KV
        # cache. A sibling still generating would hold it again, and nothing would free it.
        episode_outputs_or_errors = await asyncio.gather(
            *episode_tasks, return_exceptions=True
        )
        rollouts_or_errors: list[Rollout | BaseException] = []
        group_metrics: list[m.Metric] = []
        for episode_output_or_error in episode_outputs_or_errors:
            if isinstance(episode_output_or_error, BaseException):
                rollouts_or_errors.append(episode_output_or_error)
                continue
            rollout, rollout_metrics = episode_output_or_error
            rollouts_or_errors.append(rollout)
            group_metrics.extend(rollout_metrics)
        group = await self.make_scored_group(
            rollouts_or_errors=rollouts_or_errors, sample=sample, group_id=group_id
        )
        group.metrics = m.fold(group_metrics)
        return group

    async def score_group(
        self, rollouts: list[Rollout], env_input: object
    ) -> list[RubricOutput]:
        """Verifiers' grade for each rollout that trains (the "Failed rollouts" table in this
        folder's README), and reward None for the rest, so they don't train. A timed-out
        validation rollout also keeps its grade:

            COMPLETED, group 7         its grade
            ERROR_TIMEOUT, group 7     None        (training; the rubric skips it)
            ERROR_TIMEOUT, group -3    its grade   (validation)

        Only the rollouts that keep a grade reach the rubric.
        """
        keeps_grade = []
        for rollout in rollouts:
            # Validation groups have negative ids (`Controller`'s `_validation_group_ids`).
            is_validation_timeout = (
                rollout.group_id < 0 and rollout.status is RolloutStatus.ERROR_TIMEOUT
            )
            keeps_grade.append(
                rollout.status in _TRAINED_STATUSES or is_validation_timeout
            )
        graded_rollouts = [
            rollout for rollout, keep in zip(rollouts, keeps_grade, strict=True) if keep
        ]
        # Like `make_scored_group`: a group-level rubric never gets an empty list.
        grades = iter(
            await super().score_group(rollouts=graded_rollouts, env_input=env_input)
            if graded_rollouts
            else []
        )
        return [
            next(grades) if keep else RubricOutput(reward=None) for keep in keeps_grade
        ]

    async def _serving_env(self, env_config_json: str) -> vf.Env:
        """The env for `env_config_json`, loaded and serving from its first group on.

        Example: a run that validates on another taskset serves two envs per worker.
        """
        async with self._envs_lock:
            if env_config_json not in self._envs:
                with sl.log_trace_span("verifiers_start_env"):
                    env_config = vf.resolve_env_config(json.loads(env_config_json))
                    env = vf.load_environment(env_config)
                    # `serving()` starts the env's interception server and shared tools, which
                    # every episode of this env uses, so they live until `close`.
                    await self._serving_envs.enter_async_context(env.serving())
                self._envs[env_config_json] = env
            return self._envs[env_config_json]

    async def _run_episode(
        self,
        *,
        env: vf.Env,
        task: vf.Task,
        sampling: SamplingConfig,
        group_id: int,
        episode_index: int,
    ) -> tuple[Rollout, list[m.Metric]]:
        """Run one Verifiers episode and turn its trainable trace into a rollout, with the
        episode's index as its id. Also returns the rollout's metrics: its phase times (a rollout
        whose setup failed before its first turn included) and its number of attempts."""
        # One slot per episode, as `env.slots(task, n)` would plan: nothing couples them. The slot
        # holds the episode's traces, so they are freed when this episode returns.
        (slot,) = env.slots(task)
        # Each trace Verifiers mints, a retried attempt's included, is a routing session.
        session_ids: list[str] = []
        # The trainable agent's attempts: more than one when Verifiers retried it.
        trainable_attempt_ids: list[str] = []

        def open_session(trace: vf.Trace) -> None:
            session_ids.append(trace.id)
            if trace.agent.trainable:
                trainable_attempt_ids.append(trace.id)
            self._generation_server.open_session(session_id=trace.id, group_id=group_id)

        model_context = vf.ModelContext(
            client=self._client_config,
            model=GenerationServer.MODEL_NAME,
            sampling=vf.SamplingConfig(
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                max_tokens=sampling.max_tokens,
                seed=sampling.seed,
            ),
        )
        try:
            episode = await env.run_slot(
                slot=slot, ctx=model_context, on_trace=open_session
            )
        finally:
            # Close each session after its calls in flight end, then release it: an agent past
            # its deadline drops its request, but the generator call behind it still runs.
            sessions = {
                session_id: await self._generation_server.close_session(session_id)
                for session_id in session_ids
            }
            await asyncio.gather(
                *(
                    self._release_session_fn(routing_session_id=session_id)
                    for session_id in session_ids
                )
            )

        # One trace is one agent run, e.g. the policy and a simulated user. A harness subagent
        # (e.g. OpenCode's `task` tool) adds a branch to the same trace.
        # TODO: train several agents of one episode; that needs advantages per agent.
        trainable_traces = [trace for trace in episode.traces if trace.agent.trainable]
        if len(trainable_traces) != 1:
            raise RuntimeError(
                f"expected 1 trainable trace, got {len(trainable_traces)} "
                f"(episode error: {episode.last_error})"
            )
        (trace,) = trainable_traces
        rollout = trace_to_rollout(
            episode=episode,
            trace=trace,
            session=sessions[trace.id],
            group_id=group_id,
            rollout_id=episode_index,
        )
        # Phase times cover only the final attempt: count the attempts, so a step slowed down by
        # retries shows it.
        # TODO: count attempts per agent once several agents of one episode train: this list holds
        #   every trainable agent's traces.
        num_attempts = m.Metric(
            "verifiers/num_attempts", m.Mean(len(trainable_attempt_ids))
        )
        return rollout, [*phase_time_metrics(trace), num_attempts]


def trace_to_rollout(
    episode: vf.Episode,
    trace: vf.Trace,
    *,
    session: GenerationSession,
    group_id: int,
    rollout_id: int,
) -> Rollout:
    """One trainable agent run of `episode` as a TitanRL rollout."""
    turns = trace_to_rollout_turns(
        trace=trace, session=session, group_id=group_id, rollout_id=rollout_id
    )
    # TODO: keep the grade of a rollout with no turns, e.g. a validation timeout before the
    # policy's first reply. `RewardFromVerifiers` reads 0.0 for it today. It needs an env-reward
    # field on core `Rollout` (env rewards sit on turns), which only Verifiers would use.
    if turns:
        # The rubric reads it through `RewardFromVerifiers`.
        turns[-1].env_rewards[VERIFIERS_REWARD_KEY] = trace.reward
    status = rollout_status(episode=episode, trace=trace)
    if status not in _TRAINED_STATUSES:
        # One line per rollout that failed. `Rollout.logs["errors"]` keeps every error,
        # tracebacks included.
        last_error = trace.last_error or episode.last_error
        timing = trace.timing
        logger.warning(
            "Verifiers rollout failed: group=%d rollout=%d task=%s status=%s stop=%s turns=%d "
            "dropped=%s error=%s | boot=%.0fs setup=%.0fs agent=%.0fs scoring=%.0fs",
            group_id,
            rollout_id,
            trace.task.key,
            status.value,
            trace.stop_condition,
            trace.num_turns,
            trace.dropped,
            None if last_error is None else f"{last_error.type}: {last_error.message}",
            timing.boot.duration,
            timing.setup.duration,
            timing.agent.duration,
            timing.scoring.duration,
        )
    return Rollout(
        group_id=group_id,
        rollout_id=rollout_id,
        status=status,
        turns=turns,
        logs={
            "verifiers_trace_id": trace.id,
            "task_key": trace.task.key,
            "stop_condition": trace.stop_condition,
            "errors": [
                error.model_dump(mode="json")
                for error in (*episode.errors, *trace.errors)
            ],
        },
    )


# The statuses that train, with Verifiers' grade as their reward: the "trains" column of the
# "Failed rollouts" table in this folder's README.
_TRAINED_STATUSES = frozenset(
    {
        RolloutStatus.COMPLETED,
        RolloutStatus.TRUNCATED_MAX_TURNS,
        RolloutStatus.TRUNCATED_LENGTH,
    }
)


def rollout_status(episode: vf.Episode, trace: vf.Trace) -> RolloutStatus:
    """The status of a finished Verifiers rollout. The "Failed rollouts" table in this folder's
    README lists each status and whether it trains.
    """
    # After a retry, `errors` still holds the earlier attempts' errors; the final attempt's
    # verdict is `ok`.
    if trace.dropped is not None:
        return RolloutStatus.ERROR_INFRA
    failed = not (episode.ok and trace.ok)
    # A deadline that came with an error, e.g. a sandbox that never started, is that error.
    if trace.is_timeout and not failed:
        return RolloutStatus.ERROR_TIMEOUT
    if failed:
        # Verifiers' train client turns the renderer's refusal of an overlong prompt into a
        # ProviderError with status 400 on that call. The generation server answers every
        # failure with 500, so a 400 means that refusal.
        last_call_error = trace.calls[-1].error if trace.calls else None
        if (
            last_call_error is not None
            and last_call_error.type == "ProviderError"
            and last_call_error.status_code == 400
        ):
            return RolloutStatus.TRUNCATED_PROMPT_TOO_LONG
        # Before the policy's first sampled token, a failed harness couldn't install or launch:
        # an infra failure.
        if trace.stop_condition == "harness_error" and trace.num_turns > 0:
            return RolloutStatus.ERROR
        return RolloutStatus.ERROR_INFRA
    if trace.num_turns == 0:
        return RolloutStatus.ERROR_INFRA
    if trace.stop_condition == "max_turns":
        return RolloutStatus.TRUNCATED_MAX_TURNS
    if trace.is_truncated:
        return RolloutStatus.TRUNCATED_LENGTH
    return RolloutStatus.COMPLETED


def phase_time_metrics(trace: vf.Trace) -> list[m.Metric]:
    """Where a rollout's time went, per Verifiers phase, each as a Mean and a Max over the
    step's rollouts.

    Example, one rollout:
        phase      what runs                                        seconds   metric
        boot       the sandbox starts                               20        verifiers/boot_time_s
        setup      the task and the harness get ready in it         12        verifiers/setup_time_s
        agent      the agent's turns                                300       verifiers/agent_time_s
          model    of which model calls                             260       verifiers/model_time_s
        finalize   the task's wrap-up after the agent stops         5         verifiers/finalize_time_s
        scoring    the task's grading, e.g. its tests               40        verifiers/scoring_time_s

    A phase that never finished is left out, so it doesn't pull its Mean toward 0:

        sandbox failed to start    boot only
        agent past its deadline    agent too: Verifiers ends the span when the rollout closes.
                                   The model call the deadline cut off counts in agent_time_s
                                   but not in model_time_s: Verifiers keeps only calls that
                                   returned.
    """
    timing = trace.timing
    spans = {
        "boot": timing.boot,
        "setup": timing.setup,
        "agent": timing.agent,
        "finalize": timing.finalize,
        "scoring": timing.scoring,
    }
    # A span without an end never finished; Verifiers reports its duration as 0.
    seconds_per_phase = {
        phase: span.duration for phase, span in spans.items() if span.end
    }
    if timing.agent.end:
        seconds_per_phase["model"] = timing.agent.model.duration
    return [
        m.Metric(f"verifiers/{phase}_time_s", reduction(seconds))
        for phase, seconds in seconds_per_phase.items()
        for reduction in (m.Mean, m.Max)
    ]


def trace_to_rollout_turns(
    trace: vf.Trace,
    *,
    session: GenerationSession,
    group_id: int,
    rollout_id: int,
) -> list[RolloutTurn]:
    """Flatten a Verifiers trace into rollout turns: one turn per run of sampled tokens.

    A trace is a graph of messages; each root-to-leaf path is a branch. A compaction, which
    rewrites the history, starts a new branch:

        branch 0: user [1, 2]  A1 [3, 4]  tool [5]  A2 [6]
        branch 1: user [1, 2]  summary [7]  A3 [8]

        turn  prompt_prefix_len  prompt_delta  completion  env_messages
        0     0                  [1, 2]        [3, 4]      [tool]
        1     4                  [5]           [6]         []
        2     2                  [7]           [8]         []    <- doesn't extend turn 1:
                                                                    a new training sample

    A message several branches share becomes one turn; a branch with `trainable=False` adds
    none. Every turn gets the session's policy-version range; turn 0 also its generator metrics.
    """
    # Each node's position in the trace. Keyed by id(node), the object's identity: a pydantic
    # node can't be a dict key.
    node_index = {id(node): index for index, node in enumerate(trace.nodes)}
    branches = [branch for branch in trace.branches if branch.trainable]

    # Pass 1: one turn per run of sampled tokens, in branch order.
    turns: list[RolloutTurn] = []
    turn_by_node: dict[int, RolloutTurn] = {}
    # The last turn's prompt + completion, as (branch index, end offset in that branch).
    previous_branch, previous_end = -1, 0
    previous_token_ids = np.empty(0, dtype=np.int32)
    for branch in branches:
        token_ids, logprobs = branch.token_ids, branch.logprobs
        node_start = 0
        for node in branch.nodes:
            index = node_index[id(node)]
            if node.sampled and index not in turn_by_node:
                # `branch.logprobs` reads 0.0 for a sampled token with no logprob: refuse that.
                if len(node.logprobs) != node.mask.sum():
                    raise ValueError(
                        f"trace {trace.id}: a message has {node.mask.sum()} sampled tokens "
                        f"but {len(node.logprobs)} logprobs"
                    )
                for span_start, span_end in _true_spans(node.mask):
                    start, end = node_start + span_start, node_start + span_end
                    if branch.index == previous_branch:
                        # Same branch: the prompt continues the last turn's tokens.
                        prompt_prefix_len = previous_end
                        prompt_delta_token_ids = token_ids[previous_end:start].copy()
                    else:
                        prompt_prefix_len, prompt_delta_token_ids = split_prompt(
                            prompt=Prompt(token_ids=token_ids[:start].tolist()),
                            previous=Prompt(token_ids=previous_token_ids.tolist()),
                        )
                    turns.append(
                        RolloutTurn(
                            rollout_id=RolloutTurnID(
                                group_id=group_id,
                                rollout_id=rollout_id,
                                turn_id=len(turns),
                            ),
                            prompt_prefix_len=prompt_prefix_len,
                            prompt_delta_token_ids=prompt_delta_token_ids,
                            completion_token_ids=token_ids[start:end].copy(),
                            completion_logprobs=logprobs[start:end].copy(),
                            min_policy_version=session.min_policy_version,
                            max_policy_version=session.max_policy_version,
                            completion_message=message_to_wire(node.message),
                            metrics=list(session.metrics) if not turns else [],
                        )
                    )
                    turn_by_node[index] = turns[-1]
                    previous_branch, previous_end = branch.index, end
                    previous_token_ids = token_ids[:end]
            node_start += len(node.token_ids)

    # Pass 2: a tool or user message after a turn is that turn's env reply, attached once.
    env_message_nodes: set[int] = set()
    for branch in branches:
        replying_turn: RolloutTurn | None = None
        for node in branch.nodes:
            index = node_index[id(node)]
            if node.sampled:
                replying_turn = turn_by_node.get(index)
            elif replying_turn is not None and index not in env_message_nodes:
                env_message_nodes.add(index)
                replying_turn.env_messages.append(message_to_wire(node.message))
    return turns


def _true_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open spans of True values. Example: [F, T, T, F, T] -> [(1, 3), (4, 5)]."""
    edges = np.flatnonzero(
        np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    )
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def _bypass_proxy_for_loopback() -> None:
    """Add 127.0.0.1 and localhost to `no_proxy`, so Verifiers' calls reach the generation server.

    Why: Verifiers builds its own httpx client, which reads `http_proxy` from the environment,
    and httpx 0.28 sends even a 127.0.0.1 call to that proxy unless `no_proxy` names 127.0.0.1.

    Example: no_proxy=".corp.internal" -> no_proxy=".corp.internal,127.0.0.1,localhost"
    """
    # TODO: drop this once Verifiers' HTTP client skips the proxy for a loopback base_url.
    entries = (
        os.environ.get("no_proxy"),
        os.environ.get("NO_PROXY"),
        "127.0.0.1",
        "localhost",
    )
    no_proxy = ",".join(dict.fromkeys(entry for entry in entries if entry))
    os.environ["no_proxy"] = os.environ["NO_PROXY"] = no_proxy
