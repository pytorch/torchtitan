# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rollouter that runs Verifiers environments in TitanRL's rollout workers."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import subprocess
import uuid
from dataclasses import dataclass, replace
from typing import Any, TYPE_CHECKING

import numpy as np
from verifiers.v1.clients import ModelContext as VerifiersModelContext
from verifiers.v1.configs.client import TrainClientConfig as VerifiersTrainClientConfig
from verifiers.v1.configs.env import EnvConfig as VerifiersEnvConfig
from verifiers.v1.configs.sampling import SamplingConfig as VerifiersSamplingConfig
from verifiers.v1.configs.taskset import TasksetConfig as VerifiersTasksetConfig
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.env import Env as VerifiersEnv, RunSlot as VerifiersRunSlot
from verifiers.v1.utils.loaders import load_environment, resolve_env_config

from torchtitan.components.data.multimodal.vision_processor import VisionProcessor
from torchtitan.components.renderer import RendererConfig, RenderersConfigAdapter

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.rl.examples.verifiers.data import (
    VerifiersTaskDataset,
    VerifiersTaskSample,
)
from torchtitan.rl.examples.verifiers.generation_server import (
    GenerationServer,
    GROUP_ID_SAMPLING_PARAM,
    VerifiersGenerationMetadata,
)
from torchtitan.rl.rollout.data import IterableRolloutDataLoader
from torchtitan.rl.rollout.rollouter import BaseRolloutWorker, Rollouter
from torchtitan.rl.rollout.rubric import RewardFn
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

# The container watchdog sweeps once when its worker exits and, if the worker died, again
# this long after.
_SECOND_SWEEP_DELAY_SEC = 30


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
    """Runs Verifiers envs as TitanRL rollout groups. Verifiers runs each episode: the agent, its
    tools and the grader. TitanRL picks the tasks, serves the model calls and trains.

    One group: run_group_rollouts (from Rollouter) sends it to a random rollout worker's run_group.
    The worker starts one agent per rollout and grades it. Each agent turn: agent --HTTP--> worker,
    whose GenerationServer calls router.reserve, then generate on the generator it got.

    Example: rl_grpo_qwen35_9b_terminal_bench's 16 GPUs as SlurmLauncher would place them on two
    8-GPU hosts. With no launcher, it is one 16-GPU host: trainer GPUs 0-7, generators GPUs 8-15.

    host 0 (runs the controller)                                  host 1
    +----------------------------------------------+              +------------------------------+
    | GPU  [] [] [] [] [] [] [] []  trainer        |--weights---->| GPU  [] [] [] [] [] [] [] [] |
    | CPU  controller process: VerifiersRollouter  |              | 8 generators, 1 GPU each     |
    |      router process: InterGeneratorRouter    |--release,--->| (num_generators)             |
    |      batcher process                         | load weights |                              |
    |      16 rollout worker processes             |              |                              |
    |        (worker_pool_size). Inside each: the  |              |                              |
    |        Verifiers env and a GenerationServer  |--generate--->|                              |
    |      16 container watchdogs (`sh` processes) |              |                              |
    |      agents: a Docker container per rollout  |              |                              |
    +----------------------------------------------+              +------------------------------+

    Arrows cross hosts. weights: the tensors, over TorchStore. The others: Monarch calls.
    No TorchTitan process is pinned to cores: they share the host's CPUs.
    Agents run in the env's runtime: Docker here, remote VMs with PrimeConfig.
    Workers see no GPU (`CUDA_VISIBLE_DEVICES=""`).
    With a launcher and `workers_on_controller_host=False`, the workers, watchdogs and containers
    move to host 1. Then run_group and the calls to the router cross hosts; generate does not.
    Call order, with the release calls: torchtitan/rl/examples/verifiers/README.md.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Rollouter.Config):
        training_dataloader: IterableRolloutDataLoader.Config
        """Verifiers taskset used for training samples and environment scoring."""

        validation_dataset: VerifiersTaskDataset.Config
        """Verifiers taskset used for validation samples."""

        worker: VerifiersRolloutWorker.Config
        """How a rollout worker runs a Verifiers group; `__post_init__` fills in
        `worker.environment.taskset`."""

        def __post_init__(self) -> None:
            Rollouter.Config.__post_init__(self)
            configured_taskset = self.worker.environment.taskset
            if configured_taskset not in (
                VerifiersTasksetConfig(),
                self.training_dataloader.dataset.verifiers_taskset,
            ):
                raise ValueError(
                    "worker.environment.taskset is derived from "
                    "training_dataloader.dataset.verifiers_taskset and must not configure a "
                    "different taskset"
                )
            self.worker = replace(
                self.worker,
                environment=self.worker.environment.model_copy(
                    update={
                        "taskset": self.training_dataloader.dataset.verifiers_taskset
                    }
                ),
            )


class VerifiersRolloutWorker(BaseRolloutWorker):
    """Runs one Verifiers group in a rollout worker process. The flow: `VerifiersRollouter`."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseRolloutWorker.Config):
        environment: VerifiersEnvConfig
        """Verifiers env and agent config, e.g. `vf.SingleAgentEnvConfig(agent=...)`. Leave
        `environment.taskset` unset: `VerifiersRollouter` copies the training taskset in."""

        generation_server: GenerationServer.Config
        """Local HTTP bridge from Verifiers to TitanRL generation; each rollout worker runs one."""

        renderer_multiplex: int = 256
        """Maximum concurrent rollouts sharing one Verifiers renderer instance.

        TODO: evaluate sharing this renderer pool with TitanRL's native rollout
        path instead of keeping it specific to the Verifiers client.
        """

        def __post_init__(self) -> None:
            if self.renderer_multiplex <= 0:
                raise ValueError("renderer_multiplex must be positive")

        def to_dict(self) -> dict[str, Any]:
            # The generic dump can't serialize the pydantic env config.
            return {
                **BaseRolloutWorker.Config.to_dict(replace(self, environment=None)),
                "environment": self.environment.model_dump(mode="json"),
            }

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._config = config
        # TODO: drop this loopback bridge. `generate_fn` already calls the reserved generator
        # directly (look-aside routing); dropping the bridge needs an HTTP endpoint on each
        # generator, a reserve per model call from Verifiers' client, and another home for the
        # policy versions recorded here.
        self._generation_server: GenerationServer = config.generation_server.build()
        self._verifiers_train_client_config: VerifiersTrainClientConfig
        self._verifiers_env: VerifiersEnv
        # Holds `env.serving()` from `setup_async` to `close`. Monarch runs those two in
        # different tasks, which works while the env's serving resources are not bound to a
        # task (Verifiers 0.4.1's interception server and shared tools are not).
        self._verifiers_env_serving = contextlib.AsyncExitStack()
        self._container_watchdog: subprocess.Popen | None = None

    async def setup_async(
        self,
        *,
        tokenizer_config: HuggingFaceTokenizer.Config,
        renderer_config: RendererConfig,
        hf_assets_path: str,
        vision_processor_config: VisionProcessor.Config | None = None,
    ) -> None:
        """Start this worker's generation server, then load and serve the Verifiers env.
        Verifiers renders with its own renderer, so no tokenizer here."""
        del tokenizer_config
        if vision_processor_config is not None:
            # TODO: pass the vision processor to the renderer Verifiers builds in this worker.
            raise NotImplementedError(
                "Verifiers environments do not support image prompts yet; unset "
                "Controller.Config.vision_processor."
            )
        if not isinstance(renderer_config, RenderersConfigAdapter):
            raise ValueError(
                "Verifiers requires a renderer configured with from_renderers(...) "
                "so its client can construct the same renderer in the rollout worker"
            )
        # Verifiers' client in this worker calls its generation server on loopback. With
        # http_proxy set, httpx sends even loopback requests to the proxy unless no_proxy
        # names the host.
        # TODO: drop once Verifiers' train client ignores env proxies for a loopback base_url.
        # Keep the host's own entries from either spelling, e.g. NO_PROXY=.corp.internal.
        entries = (
            os.environ.get("no_proxy"),
            os.environ.get("NO_PROXY"),
            "127.0.0.1",
            "localhost",
        )
        no_proxy = ",".join(dict.fromkeys(entry for entry in entries if entry))
        os.environ["no_proxy"] = no_proxy
        os.environ["NO_PROXY"] = no_proxy
        await self._generation_server.start()
        self._verifiers_train_client_config = VerifiersTrainClientConfig(
            base_url=self._generation_server.base_url,
            # The local endpoint does not authenticate. An empty environment
            # variable name makes Verifiers send its required "EMPTY" value.
            api_key_var="",
            renderer=renderer_config.get_renderers_config(),
            multiplex=self._config.renderer_multiplex,
            renderer_model_name=hf_assets_path,
        )
        # One VF_RUN_ID per worker: "torchtitan-<uuid>", or "<launcher's VF_RUN_ID>-<uuid>".
        # 1. Verifiers labels each container it starts `verifiers.run=$VF_RUN_ID`.
        # 2. So the watchdog removes exactly this worker's containers when the worker dies.
        # 3. Side effect: Verifiers' run-wide creation-rate limits (e.g. `PrimeConfig.creates_per_min`)
        #    apply per worker.
        # TODO: drop the side effect once Verifiers keeps the container label and the limiter scope apart.
        run_id_prefix = os.environ.get("VF_RUN_ID", "torchtitan")
        os.environ["VF_RUN_ID"] = f"{run_id_prefix}-{uuid.uuid4().hex}"
        self._container_watchdog = _start_container_watchdog(os.environ["VF_RUN_ID"])
        # resolve_env_config re-validates `environment` as the config class of the Env that
        # the taskset's package exports (SingleAgentEnv's if it exports none).
        self._verifiers_env = load_environment(
            resolve_env_config(self._config.environment)
        )
        # The env's serving resources (the interception server its agents call, shared
        # tools) live until `close`, as in Verifiers' own runner.
        await self._verifiers_env_serving.enter_async_context(
            self._verifiers_env.serving()
        )

    async def close(self) -> None:
        """Stop serving the env, then the generation server and the container watchdog."""
        env_stopped = False
        try:
            await self._verifiers_env_serving.aclose()
            env_stopped = True
        finally:
            await self._generation_server.close()
            if self._container_watchdog is not None:
                # Once the env stopped, no `docker run` is in flight: "clean" skips the delayed
                # second sweep, so the `wait` below returns after one sweep. If `aclose()` raised,
                # a `docker run` may still be in flight, so the watchdog sweeps twice.
                if env_stopped:
                    self._container_watchdog.stdin.write(b"clean\n")
                self._container_watchdog.stdin.close()
                try:
                    await asyncio.to_thread(self._container_watchdog.wait, 60)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "The container watchdog of run %s is still sweeping after 60 s; "
                        "it goes on after this worker exits.",
                        os.environ["VF_RUN_ID"],
                    )

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
        if not isinstance(sample, VerifiersTaskSample):
            raise TypeError("Verifiers requires a VerifiersTaskSample")
        # Each group replaces this one slot. That is safe because every group's generate_fn routes the
        # same way; keep per-group state out of it.
        self._generation_server.set_generate_fn(generate_fn)
        # Rebuild the controller's task from its data, as the taskset's `load` builds it.
        task_type = self._verifiers_env.taskset.task_type()
        task = task_type(
            task_type.data_type().model_validate(sample.verifiers_task_data),
            self._verifiers_env.taskset.config.task,
        )
        results = await asyncio.gather(
            *(
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
                for rollout_id, slot in enumerate(
                    self._verifiers_env.slots(task, group_size)
                )
            ),
            return_exceptions=True,
        )
        return await self.make_scored_group(results, sample, group_id=group_id)

    async def _run_single_rollout(
        self,
        *,
        release_session_fn: ReleaseSessionFn,
        slot: VerifiersRunSlot,
        sampling: SamplingConfig,
        group_id: int,
        rollout_id: int,
    ) -> Rollout:
        """Run one Verifiers episode, release its sessions, and convert its trace to a rollout."""
        # One run_slot executes a complete episode. The harness owns the multi-turn
        # model/tool loop and calls the generation server once per generation; run_slot
        # returns only after the episode stops.
        verifiers_episode = await self._verifiers_env.run_slot(
            slot,
            VerifiersModelContext(
                client=self._verifiers_train_client_config,
                model=self._generation_server.model_id,
                sampling=VerifiersSamplingConfig(
                    temperature=sampling.temperature,
                    top_p=sampling.top_p,
                    max_tokens=sampling.max_tokens,
                    seed=sampling.seed,
                    **{GROUP_ID_SAMPLING_PARAM: group_id},
                ),
            ),
        )
        # `run_slot` returned, so this rollout starts no new calls. For each trace, release its routing
        # session (Verifiers sends the trace id as X-Session-ID) and take its generation metadata,
        # so another agent's trace (a judge, a simulated user) leaves nothing on the server.
        # TODO: an attempt Verifiers retried has no trace here, so its held KV waits for the
        # group's release, its routing pin for the router's `max_sessions` LRU, and its generation
        # metadata and request counts stay on the generation server (kept per session id). The
        # generation server could record each rollout's session ids and release those too.
        generation_metadata_by_session = {
            trace.id: self._generation_server.pop_generation_metadata(trace.id)
            for trace in verifiers_episode.traces
        }

        async def release(session_id: str) -> None:
            try:
                await release_session_fn(routing_session_id=session_id)
            except Exception:
                # Not fatal: `release_groups` frees the group's KV when the group ends, and
                # `max_sessions` evicts the routing pin.
                logger.exception("releasing routing session %s failed", session_id)

        await asyncio.gather(*(release(trace.id) for trace in verifiers_episode.traces))
        traces = [trace for trace in verifiers_episode.traces if trace.agent.trainable]
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
        status = self.rollout_status(verifiers_episode=verifiers_episode, trace=trace)
        if not turns:
            # Verifiers records an overlong first prompt as a failed model call; vLLM, the
            # renderer and the generation server all word it "maximum context length".
            first_prompt_too_long = any(
                call.error is not None
                and "maximum context length" in call.error.message
                for call in trace.calls
            )
            if first_prompt_too_long:
                # The harness builds the first prompt during the episode, so the dataset can't
                # be filtered for it; flag it here as a data problem, not a policy one.
                # Not every task has a name; every task has an index.
                task = slot.task.data.name or slot.task.data.idx
                logger.warning(
                    "rollout group=%d/rollout=%d: the first prompt of task %s does not fit "
                    "the context; marking ERROR",
                    group_id,
                    rollout_id,
                    task,
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
            # Verifiers' int32 array; `split_prompt` compares lists.
            token_ids = branch.token_ids.tolist()
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


def _start_container_watchdog(run_id: str) -> subprocess.Popen:
    """Start a process that removes the containers labelled `verifiers.run=<run_id>` once this
    process exits, also on SIGKILL: the kernel then closes this process's end of the pipe.

    A killed worker runs no cleanup of its own: Monarch runs `__cleanup__` only on a graceful
    stop, and Python can't handle SIGTERM off the main thread, where actor code runs.

    Example: a rollout worker killed mid-episode -> its watchdog runs `docker rm -f` on the
    episode's containers, then sweeps again `_SECOND_SWEEP_DELAY_SEC` later. A clean
    `close()` -> one sweep.

    Docker only: without `docker` it removes nothing. Other sandbox providers reap with their own
    lease or heartbeat.
    """
    # TODO: drop this watchdog if Verifiers brings back a reaper for killed processes
    # (0.4.0's env server had one: `_arm_teardown` in `verifiers/v1/serve/pool.py`).
    # TODO: a kill of the whole process tree at once (e.g. its cgroup) takes the watchdog too.
    # TODO: Compose projects (`vf-<uuid>`) carry no run label, so the watchdog misses them;
    # no Terminal-Bench 2.1 or TMax task uses Compose.
    # The second sweep catches a container whose `docker run` was still in flight when the
    # worker died: it runs only on EOF without `close()`'s "clean" line. `docker` errors go
    # to the worker's stderr.
    script = (
        "read -r how; command -v docker >/dev/null || exit 0; "
        'sweep() { ids=$(docker ps -aq --filter "label=verifiers.run=$1"); '
        '[ -z "$ids" ] || docker rm -f $ids >/dev/null; }; '
        'sweep "$1"; [ "$how" = clean ] || { sleep "$2"; sweep "$1"; }'
    )
    # Its own session, so a signal sent to the worker's process group doesn't stop it.
    return subprocess.Popen(
        [
            "sh",
            "-c",
            script,
            "container-watchdog",
            run_id,
            str(_SECOND_SWEEP_DELAY_SEC),
        ],
        stdin=subprocess.PIPE,
        start_new_session=True,
    )


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
