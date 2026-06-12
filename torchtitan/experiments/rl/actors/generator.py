# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import enum
import gc
import logging
import math
import os
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import torchstore as ts
from monarch.actor import Actor, current_rank, endpoint
from monarch.rdma import is_rdma_available
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.config import (
    CompileConfig,
    Configurable,
    DebugConfig,
    ParallelismConfig,
)
from torchtitan.distributed.utils import set_batch_invariance
from torchtitan.experiments.rl.models.vllm_registry import (
    registry_to_vllm,
    TORCHTITAN_CONFIG_FORMAT,
)
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.types import Completion
from torchtitan.models.common.attention import FlexAttention, VarlenAttention
from torchtitan.observability import structured_logger as sl
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools.logging import init_logger
from torchtitan.tools.utils import has_cuda_capability
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.config import AttentionConfig, CompilationConfig
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.attention.backends.registry import AttentionBackendEnum

logger = logging.getLogger(__name__)


def _prepare_generation_request_metrics(
    request_output: RequestOutput, *, prefix: str
) -> list[m.Metric]:
    """Prepare vLLM metrics from a RequestOutput.

    For `add_request` call, vLLM returns RequestOutput carrying
    a single `RequestStateStats` on `.metrics` field.

    Caveat under `SamplingParams.n > 1`: vLLM stores one `RequestStateStats`
    per child request; the parent output exposes the **last-finishing**
    child's timeline. `arrival_time` is shared across siblings, but
    [`queued_ts`, `scheduled_ts`, `first_token_ts`, `last_token_ts`,
    `num_generation_tokens`] describe one specific child — not an aggregate,
    not the first sibling's. The other `n-1` siblings' stats are dropped by
    vLLM at ``output_processor._finish_request``.
    """

    # TODO: Per-request fields here come from RequestOutput.metrics
    # (RequestStateStats). Engine-aggregate stats, such as KV-cache usage,
    # prefix-cache hit rate, preemptions, and batch occupancy, live in
    # SchedulerStats / IterationStats and require registering a
    # vllm.v1.metrics.loggers.StatLoggerBase via
    # LLMEngine.from_engine_args(..., stat_loggers=[...]).

    metric_values: dict[str, float] = {}
    if request_output.num_cached_tokens is not None:
        metric_values[f"{prefix}/num_cached_tokens"] = request_output.num_cached_tokens

    stats = request_output.metrics
    if stats is not None:
        metric_values[f"{prefix}/queue_time_ms"] = (
            stats.scheduled_ts - stats.queued_ts
        ) * 1000

        if stats.num_generation_tokens > 0:
            metric_values[f"{prefix}/time_to_first_token_ms"] = (
                stats.first_token_latency * 1000
            )
            metric_values[f"{prefix}/prefill_time_ms"] = (
                stats.first_token_ts - stats.scheduled_ts
            ) * 1000

        if stats.num_generation_tokens > 1:
            first_to_last_token_ms = (stats.last_token_ts - stats.first_token_ts) * 1000
            metric_values[f"{prefix}/decode_time_ms"] = first_to_last_token_ms
            metric_values[
                f"{prefix}/inter_token_latency_ms"
            ] = first_to_last_token_ms / (stats.num_generation_tokens - 1)

    # Emit each value with both Mean and Max aggregators.
    return [
        metric
        for key, value in metric_values.items()
        for metric in (m.Metric(key, m.Mean(value)), m.Metric(key, m.Max(value)))
    ]


@dataclass(kw_only=True, slots=True)
class VLLMCudagraphConfig:
    """CUDA graph capture settings for the vLLM inference engine.

    torch.compile is configured separately via ``CompileConfig`` at the
    ``RLTrainer`` level, shared by both trainer and generator.  Only CUDA
    graph capture, which is vLLM-specific, is controlled here.

    When enabled, vLLM captures the forward pass as a single CUDA graph
    ("full" mode).  "piecewise" modes are intentionally excluded: they
    require vLLM's whole-model torch.compile to split the graph around
    non-capturable ops, which conflicts with per-layer compile.
    """

    enable: bool = True
    """Whether to enable CUDA graph capture (vLLM "full" mode)."""

    # TODO: Validate CUDA graph capture with MoE / Expert Parallelism.
    # MoE routing produces dynamic shapes that may conflict with full
    # CUDA graph capture despite being torch.compile-compatible
    # post https://github.com/pytorch/torchtitan/pull/3142

    # TODO: Explore applying CUDA graph capture on the torchtitan trainer
    # side as well (not just the vLLM generator).
    # https://github.com/pytorch/torchtitan/issues/3175

    def get_vllm_compilation_config(
        self, *, max_num_seqs: int
    ) -> CompilationConfig | None:
        """Build a vLLM ``CompilationConfig``, or return ``None`` when
        CUDA graphs are disabled.

        ``max_num_seqs`` determines CUDA graph capture sizes: powers of
        2 from 1 up to ``max_num_seqs``, plus ``max_num_seqs`` itself
        if it isn't already a power of 2.
        """
        if not self.enable:
            return None
        if max_num_seqs <= 0:
            raise ValueError(f"max_num_seqs must be positive, got {max_num_seqs}")
        sizes = [1 << i for i in range(int(math.log2(max_num_seqs)) + 1)]
        if max_num_seqs not in sizes:
            sizes.append(max_num_seqs)
        return CompilationConfig(
            cudagraph_mode="full",
            mode=0,
            cudagraph_capture_sizes=sorted(sizes),
        )


@dataclass(kw_only=True, slots=True)
class SamplingConfig:
    """Sampling parameters passed to vLLM's SamplingParams."""

    temperature: float = 0.8
    """Sampling temperature. 0.0 = greedy, higher = more random."""

    top_p: float = 0.95
    """Nucleus sampling threshold."""

    max_tokens: int = 100
    """Maximum number of tokens to generate per completion."""


class VLLMGenerator(Actor, Configurable):
    """vLLM engine to drive concurrent `generate` calls through one SPMD engine loop.

    The controller fires independent calls (`generate`, `pull_model_state_dict`, `close`).
    Rank 0 processes them, enqueue a `LoopDecision` and awaits a future. One background `_engine_loop` per rank
    consumes the queue and executes the action. Rank 0 resolves each future when its request finishes and return
    the result back to the controller.

    Notice that vLLM `engine.step`, which is a TP collective, and the request-intake are decoupled, so a new request
    can join mid-flight, instead of waiting for the current batch to drain.

    One loop iteration, TP=2, after the controller fired generate(prompt_0) and generate(prompt_1):

        # request intake in `generate` takes a prompt, puts in a queue, releases control back to the controller
        generate(prompt_0): enqueue prompt_0, await gen_future_0   ┐ rank 0 owns the queue + futures
        generate(prompt_1): enqueue prompt_1, await gen_future_1   ┘ (other ranks are no-op)

        # meanwhile, the engine-loop, which is its own coroutine, is continuously running.
        rank 0   _decide_next_action -> LoopDecision(STEP, [prompt_0, prompt_1])─┐  broadcast_object_list (gloo)
        rank 1   (blocked inside the broadcast) ─────────────────────────────────┘  decision from rank0 is broadcast

        # The loop now executes the decision. In this example: STEP.
        ALL      add_request(prompt_0), add_request(prompt_1)
        ALL      engine.step() * max_engine_steps_between_decisions # N step burst before a new decision

        # resolve the future, waking up `generate` so it returns the result to the controller.
        # Note that prompt_1 can be done before prompt_0. The result is per request, not per batch.
        rank 0   _process_finished_requests -> prompt_1 done? gen_future_1.set_result(Completion)
        rank 1   _process_finished_requests -> no-op (holds no futures)

    A weight sync rides the same loop: `pull_model_state_dict` queues a `LoopDecision(LoopAction.PULL_MODEL_STATE_DICT)` applied
    between step bursts. The engine does NOT drain in-flight requests first ("hotswap"). This behavior can be changed
    on the controller side, by blocking new requests until the engine is drained.

    Args:
        config: Generator-specific configuration.
        model_spec: TorchTitan model specification.
        model_path: Path to the HF model checkpoint.
        compile_config: Per-layer torch.compile config shared with the
            trainer so both sides compile identically.
        max_num_seqs: vLLM's max concurrent sequences (KV budget + CUDA-graph sizes).
        output_dir: Structured-logger output directory.
        stop_token_ids: Renderer role-boundary stop tokens injected by the controller.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Generator actor configuration.
        TODO: Expose a EngineConfig field to passing config to vLLM Engine"""

        parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
        """Parallelism configuration for the vLLM engine."""

        sampling: SamplingConfig = field(default_factory=SamplingConfig)
        """Default sampling parameters for generation."""

        model_dtype: str = "bfloat16"
        """Data type for model weights, passed directly to vLLM (auto, float16, bfloat16, float32)."""

        gpu_memory_limit: float = 0.9
        """Fraction of GPU memory to use for the vLLM engine (0.0 to 1.0)."""

        cudagraph: VLLMCudagraphConfig = field(default_factory=VLLMCudagraphConfig)
        """CUDA graph capture settings for the vLLM engine."""

        checkpoint: CheckpointManager.Config = field(
            default_factory=CheckpointManager.Config
        )
        """Controls whether the vLLM wrapper loads initial HF weights.
        In the RL loop this should stay disabled (default ``enable=False``)
        because weights arrive from TorchStore. For standalone inference,
        set ``enable=True`` and ``initial_load_in_hf=True``."""

        debug: DebugConfig = field(default_factory=DebugConfig)
        """Debug and determinism settings."""

        max_engine_steps_between_decisions: int = 16
        """Controls how many `engine.step()` calls the `engine_loop` performs before processing a new decision.
        Every generation call is queued for execution by the `engine_loop`. A higher value enables buffering
        of more requests to avoid a prefill between every engine decode step, which is inefficient."""

        # TODO: check if we should put these under WeightSyncConfig
        reset_prefix_cache_on_weight_sync: bool = True
        """Drop the prefix cache when weights change so new requests don't reuse KV computed under the old
        weights. vLLM only clears it while the engine is idle (true under sync training)."""

        reset_running_requests_on_weight_sync: bool = False
        """Affects requests ALREADY running at the pull: preempts them and recomputes their KV under
        the new weights. No effect under strict-drain (engine idle at pull time); async hot-swap only."""

        def __post_init__(self):
            # VLLMGenerator only supports TP. vLLM handles its own parallelism;
            # we only apply TP via the core parallelize function.
            p = self.parallelism
            if p.data_parallel_replicate_degree != 1:
                raise ValueError(
                    f"Generator does not support data parallel replication, "
                    f"got dp_replicate={p.data_parallel_replicate_degree}"
                )
            if p.pipeline_parallel_degree > 1:
                raise ValueError(
                    f"Generator does not support pipeline parallelism, "
                    f"got pp={p.pipeline_parallel_degree}"
                )
            if p.context_parallel_degree > 1:
                raise ValueError(
                    f"Generator does not support context parallelism, "
                    f"got cp={p.context_parallel_degree}"
                )
            if p.expert_parallel_degree > 1:
                raise ValueError(
                    f"Generator does not support expert parallelism, "
                    f"got ep={p.expert_parallel_degree}"
                )
            if p.enable_sequence_parallel:
                raise ValueError(
                    "Generator does not support sequence parallelism: "
                    "spmd_types erasure mode requires sequence length to be "
                    "evenly divisible by TP, which doesn't hold for inference "
                    "(uneven batches). Set enable_sequence_parallel=False."
                )
            if not p.disable_loss_parallel:
                raise ValueError(
                    "Generator requires disable_loss_parallel=True, "
                    f"got disable_loss_parallel={p.disable_loss_parallel}"
                )

            if (
                self.debug.batch_invariant
                and not self.reset_prefix_cache_on_weight_sync
            ):
                raise ValueError(
                    "batch_invariant requires reset_prefix_cache_on_weight_sync=True so a stale prefix "
                    "cache from old weights can't break determinism"
                )
            if (
                self.reset_running_requests_on_weight_sync
                and not self.reset_prefix_cache_on_weight_sync
            ):
                raise ValueError(
                    "reset_running_requests_on_weight_sync requires "
                    "reset_prefix_cache_on_weight_sync=True (it only matters as part of resetting the cache)"
                )

    def __init__(
        self,
        config: Config,
        *,
        model_spec: ModelSpec,
        model_path: str,
        compile_config: CompileConfig,
        max_num_seqs: int,
        output_dir: str,
        stop_token_ids: list[int],
    ):
        init_logger()
        sl.init_structured_logger(
            source="rl_generator",
            output_dir=output_dir,
            rank=current_rank().rank,
            enable=config.debug.enable_structured_logging,
        )
        sl.log_trace_instant("structured_logger_started")

        self.config = config
        self.model_spec = model_spec

        # max_num_seqs controls vLLM's maximum batch dimension: it sets
        # the upper bound for concurrent sequences, determines KV-cache
        # block allocation (and therefore GPU memory usage), and bounds
        # the CUDA graph capture sizes.  Always computed by the caller
        # (RLTrainer) as num_groups_per_rollout_batch * group_size.
        self._max_num_seqs = max_num_seqs

        # Renderer role-boundary stop tokens (e.g. Qwen3 `<|im_end|>`), injected by the
        # controller;
        self._stop_token_ids = stop_token_ids

        # Register TorchTitan model + parser with vLLM
        registry_to_vllm(
            model_spec,
            parallelism=config.parallelism,
            compile_config=compile_config,
            checkpoint_config=config.checkpoint,
        )

        # Set vLLM environment variables from config before any vLLM initialization
        inner_attn = model_spec.model.layers[0].attention.inner_attention
        assert isinstance(
            inner_attn,
            (VarlenAttention.Config, FlexAttention.Config),
        ), "Only varlen and flex attention backends are allowed."

        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
        set_batch_invariance(config.debug.batch_invariant)

        self._set_determinism(config.debug)

        self.model_path = model_path

        # Build vLLM engine
        engine_kwargs = dict(
            # ``model`` is the path to the HF checkpoint directory. The
            # config is sourced from torchtitan's ModelSpec via
            # ``config_format=TORCHTITAN_CONFIG_FORMAT`` (no config.json
            # read), but vLLM still uses this path to locate the
            # tokenizer assets and the safetensors weight shards.
            model=model_path,
            trust_remote_code=True,
            # Use the torchtitan custom config parser (registered by
            # registry_to_vllm above). It builds PretrainedConfig from
            # ModelSpec instead of reading config.json from disk.
            config_format=TORCHTITAN_CONFIG_FORMAT,
            dtype=config.model_dtype,
            tensor_parallel_size=config.parallelism.tensor_parallel_degree,
            # Monarch already spawned TP workers via proc mesh. "external_launcher"
            # tells vLLM to run one worker per process (no subprocess spawning)
            distributed_executor_backend="external_launcher",
            gpu_memory_utilization=config.gpu_memory_limit,
            enforce_eager=not config.cudagraph.enable,
            attention_config=AttentionConfig(
                backend=(
                    AttentionBackendEnum.FLEX_ATTENTION
                    if isinstance(inner_attn, FlexAttention.Config)
                    else AttentionBackendEnum.CUSTOM
                ),
            ),
            # Enables RequestOutput.metrics, so generator metrics can be returned
            disable_log_stats=False,
        )
        engine_kwargs["max_model_len"] = model_spec.model.max_seq_len
        engine_kwargs["max_num_seqs"] = self._max_num_seqs
        # Continuous batching requires FCFS scheduling: admission order must equal the
        # broadcast order on every rank
        engine_kwargs["scheduling_policy"] = "fcfs"
        # FA2 requires block_size to be a multiple of 256
        if not has_cuda_capability(9, 0):
            engine_kwargs["block_size"] = 256
        vllm_compilation_config = config.cudagraph.get_vllm_compilation_config(
            max_num_seqs=self._max_num_seqs,
        )
        if vllm_compilation_config is not None:
            engine_kwargs["compilation_config"] = vllm_compilation_config
        if config.debug.seed is not None:
            engine_kwargs["seed"] = config.debug.seed
        engine_args = EngineArgs(**engine_kwargs)

        with sl.log_trace_span("vllm_init"):
            logger.info("Initializing LLMEngine from EngineArgs...")
            self._engine = LLMEngine.from_engine_args(engine_args)
            logger.info("vLLM rollout engine initialized")

        self.policy_version = 0

        # --- Continuous-batching state (see the class docstring) ---
        self._rank = dist.get_rank()
        self._broadcast_group = dist.new_group(backend="gloo")  # for LoopDecisions
        self._engine_loop_condition = (
            asyncio.Condition()
        )  # Signals to wake up when there is work

        # Engine-loop INBOX (rank 0): requests the controller submits; the loop reads them to decide.
        self._queued_generation_requests: list[GenerationRequest] = []
        self._model_state_dict_pull_request: ModelStateDictPullRequest | None = None
        self._close_request: CloseRequest | None = None

        # RANK-0 OUTBOX: futures the loop resolves so the awaiting endpoint returns.
        # (close has no future here: its completion handle is `_engine_loop_task`.)
        self._generation_futures: dict[str, GenerationFuture] = {}
        self._pull_model_state_dict_future: asyncio.Future[int] | None = None

        # Background asyncio.Task running _engine_loop; None until the first generate/pull starts it.
        self._engine_loop_task: asyncio.Task | None = None

        logger.info("Generator initialized with vLLM engine")

    @staticmethod
    def _set_determinism(debug: DebugConfig) -> None:
        """Apply deterministic flags for the generator.

        The generator doesn't use torchtitan's ParallelDims, so we apply
        the deterministic flags directly instead of using set_determinism().
        """
        if debug.deterministic:
            torch.use_deterministic_algorithms(
                True, warn_only=debug.deterministic_warn_only
            )
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        if debug.seed is not None:
            torch.manual_seed(debug.seed)

    def _get_model(self):
        """Access the model from the vLLM engine.
        Returns a VLLMModelWrapper instance.
        """
        return self._engine.model_executor.driver_worker.get_model()

    @endpoint
    async def sync_log_step(self, step: int, relative_step: int | None = None) -> None:
        """Sync the structured-logger step counter from the controller."""
        sl.set_step(step, relative_step=relative_step)

    @endpoint
    @sl.log_trace_span("generate")
    async def generate(
        self,
        prompt_token_ids: list[int],
        *,
        request_id: str,
        sampling_config: SamplingConfig | None = None,
        metrics_prefix: str = "generator",
    ) -> Completion | None:
        """Generates one completion for one prompt.

        Returns the `Completion` on rank 0 and `None` on followers. The completion carries its
        own per-generation metrics (`Completion.metrics`), which the controller attaches to the
        rollout turn.

        Args:
            prompt_token_ids: One tokenized prompt `[token_ids]`.
            request_id: Unique id for this request, echoed on the `Completion`.
            sampling_config: Optional per-call override for the generator's
                default SamplingConfig.
            metrics_prefix: Namespace prepended to every metric key on the returned
                `Completion` (default ``"generator"``). Callers that need to keep streams
                separate, e.g. ``"validation/generator"``, can override it.

        Example:

            completion = await generator.generate.call(
                [1, 2, 3], request_id="step=3/group=0/sample=0/turn=0",
            )
            # rank 0 -> Completion(token_ids=[...], metrics=[Metric("generator/queue_time_ms", ...)]);
            # followers -> None
        """

        # Starting requires asyncio, which isn't available in the sync __init__.
        # Start on first call; no-op after.
        await self._ensure_engine_loop()

        # Only rank 0 owns the queue + futures moving forward.
        if self._rank != 0:
            return None

        sampling = (
            sampling_config if sampling_config is not None else self.config.sampling
        )

        # `_engine_loop_condition` wakes the engine loop, if asleep, when a new request is added.
        async with self._engine_loop_condition:
            if request_id in self._generation_futures:
                raise ValueError(f"request_id {request_id!r} is already in flight")

            # A placeholder future for the engine loop to resolve with this request's Completion.
            generation_future: asyncio.Future[
                Completion
            ] = asyncio.get_running_loop().create_future()

            # Register the future before enqueueing; the engine loop resolves it.
            self._generation_futures[request_id] = GenerationFuture(
                future=generation_future, metrics_prefix=metrics_prefix
            )

            # Add the request to the queue; the engine loop will admit + process it.
            self._queued_generation_requests.append(
                GenerationRequest(
                    request_id=request_id,
                    prompt_token_ids=prompt_token_ids,
                    sampling=sampling,
                )
            )
            # Wakes the engine loop only if it is idle in `_decide_next_action`.
            self._engine_loop_condition.notify()

        # Await outside the lock so other generate / pull calls can proceed meanwhile.
        return await generation_future

    async def _ensure_engine_loop(self) -> None:
        """Start the single background engine loop on first use (idempotent); runs until `close()`."""
        if self._engine_loop_task is None:
            self._engine_loop_task = asyncio.create_task(self._engine_loop())

    @sl.log_trace_span("engine_loop")
    async def _engine_loop(self) -> None:
        """Non-stop loop running on all ranks to produce new tokens.

        Rank 0 decides a `LoopDecision` and broadcasts it; ALL ranks apply it in
        lockstep until CLOSE. On crash, fail every outstanding future so callers don't hang.

        `_decide_next_action` is consulted once every `max_engine_steps_between_decisions` steps (a burst),
        so new requests buffer and prefill together instead of on every step.

        Example:
            max_engine_steps_between_decisions = 16
            check `_decide_next_action` --> "STEP"         --> run engine.step 16 times
            check `_decide_next_action` --> "PULL_MODEL_STATE_DICT" --> run `_pull_model_state_dict`
            check `_decide_next_action` --> "STEP"         --> run engine.step 16 times
            check `_decide_next_action` --> "CLOSE"        --> stop
        """
        try:
            while True:
                # Rank 0 decides next decision; followers pass None and learn from the broadcast.
                decision = await self._decide_next_action() if self._rank == 0 else None

                # Barrier(gloo, CPU): Ship rank 0's decision (incl. prompts) to every TP rank via gloo/CPU, off the
                # NCCL stream. broadcast_object_list mutates a list in place; `to_thread` runs the
                # blocking call in a worker so the event loop can still serve generate/pull/close.
                # TODO(perf): overlap this broadcast with the step burst (pipeline the next decision) so
                # gloo transfer hides behind GPU compute.
                # TODO(perf): swap broadcast_object_list (double serialization of pickle+broadcast) for a byte
                # broadcast, to cut pickle overhead. i.e. serialize the decision to bytes on rank 0, then broadcast.
                # TODO: Revisit when enabling DP>1
                decision_broadcast_container = [decision] if self._rank == 0 else [None]
                await asyncio.to_thread(
                    dist.broadcast_object_list,
                    decision_broadcast_container,
                    src=0,
                    group=self._broadcast_group,
                    device=torch.device("cpu"),
                )
                # get rank 0's broadcasted decision
                decision = decision_broadcast_container[0]  # [num_ranks]

                if decision.action is LoopAction.CLOSE:
                    return

                if decision.action is LoopAction.PULL_MODEL_STATE_DICT:
                    await self._pull_model_state_dict(decision.pull_version)
                    continue  # back to the start for the next decision

                if decision.action is LoopAction.STEP:
                    # Add any newly-queued requests; all ranks add the identical set in FCFS order.
                    if decision.requests:
                        # render_cmpl is vLLM's input pipeline (tokenize is a no-op for tokenized prompts);
                        # the high-level entry stays resilient to vLLM internals vs vllm.inputs.tokens_input.
                        engine_inputs = self._engine.renderer.render_cmpl(
                            [
                                {"prompt_token_ids": request.prompt_token_ids}
                                for request in decision.requests
                            ]
                        )
                        for request, engine_input in zip(
                            decision.requests, engine_inputs, strict=True
                        ):
                            self._engine.add_request(
                                request_id=request.request_id,
                                prompt=engine_input,
                                params=self._build_sampling_params(request.sampling),
                            )
                            # Stamp the admission (sampling) version. A pull takes priority over
                            # STEP in `_decide_next_action`, so by here `policy_version` already
                            # reflects any just-applied swap. Only rank 0 holds the futures.
                            # TODO(async): record exact in-turn swap boundaries (a mid-decode pull
                            #   splits this into >1 interval); FINAL_ONLY output hides per-step counts.
                            if self._rank == 0:
                                self._generation_futures[
                                    request.request_id
                                ].version_intervals = [(0, self.policy_version)]

                # Barrier (NCCL): engine.step() runs SPMD in lockstep.
                # The step burst `max_engine_steps_between_decisions` gives the generator time to buffer
                # new requests and avoid a prefill in between every engine decode step, which is inefficient.
                with sl.log_trace_span("vllm_engine_step_burst"):
                    for _ in range(self.config.max_engine_steps_between_decisions):
                        if not self._engine.has_unfinished_requests():
                            break
                        with torch.no_grad():
                            with sl.log_trace_span("vllm_engine_step"):
                                request_outputs = self._engine.step()
                        self._process_finished_requests(request_outputs)
                        await asyncio.sleep(0)  # let pending generate() calls enqueue

        except Exception as exc:
            logger.exception("engine loop crashed; failing all outstanding futures")
            self._fail_outstanding_futures(exc)
            raise

    async def _decide_next_action(self) -> LoopDecision:
        """RANK 0: picks the next action. Sleeps until there is something to do."""

        # `self._engine_loop_condition.wait_for` blocks until there is work; `notify()` rechecks
        # the predicate. In-flight requests keep the predicate true, so they need no notify.
        async with self._engine_loop_condition:
            await self._engine_loop_condition.wait_for(
                lambda: self._close_request is not None
                or self._model_state_dict_pull_request is not None
                or self._queued_generation_requests
                or self._engine.has_unfinished_requests()
            )

            if self._close_request is not None:
                return LoopDecision(action=LoopAction.CLOSE, requests=[])

            # A weight pull takes priority over admitting new requests.
            if self._model_state_dict_pull_request is not None:
                return LoopDecision(
                    action=LoopAction.PULL_MODEL_STATE_DICT,
                    requests=[],
                    pull_version=self._model_state_dict_pull_request.version,
                )

            # STEP: admit whatever is queued (may be empty -> just keep stepping in-flight work).
            requests, self._queued_generation_requests = (
                self._queued_generation_requests,
                [],
            )
            return LoopDecision(action=LoopAction.STEP, requests=requests)

    def _process_finished_requests(self, request_outputs: list[RequestOutput]) -> None:
        """RANK 0: resolve each finished request's future with its `Completion` (metrics included)."""
        if self._rank != 0:
            return  # other ranks hold no futures

        for request_output in request_outputs:
            # We enforce n=1 in sampling params -> exactly one CompletionOutput per finished request
            # Here we just sanity check it (a single engine.step may still finish several requests).
            if len(request_output.outputs) != 1:
                raise ValueError(
                    f"expected n=1 (one sample per request), got "
                    f"{len(request_output.outputs)} for {request_output.request_id}"
                )

            # in flight when this one finished (includes itself; counted before the pop)
            inflight_requests_at_completion = float(len(self._generation_futures))
            generation_future = self._generation_futures.pop(request_output.request_id)

            # get logprobs
            completion_output = request_output.outputs[0]
            token_logprobs = [
                next(iter(logprob_dict.values())).logprob
                for logprob_dict in completion_output.logprobs
            ]

            # prepare metrics
            metrics_prefix = generation_future.metrics_prefix
            metrics = _prepare_generation_request_metrics(
                request_output, prefix=metrics_prefix
            )

            for metric_type in [m.Max, m.Mean]:
                metrics.append(
                    m.Metric(
                        f"{metrics_prefix}/inflight_requests_at_completion",
                        metric_type(inflight_requests_at_completion),
                    )
                )
            # Attribute the completion to the version it was ADMITTED (sampled) at, not the
            # current `self.policy_version` (a mid-flight hotswap may have advanced it): the
            # off-policy filter must see the oldest version the tokens were sampled at.
            admission_version = generation_future.version_intervals[0][1]
            generation_future.future.set_result(
                Completion(
                    policy_version=admission_version,
                    request_id=request_output.request_id,
                    token_ids=list(completion_output.token_ids),
                    token_logprobs=token_logprobs,
                    finish_reason=completion_output.finish_reason,
                    metrics=metrics,
                    version_intervals=generation_future.version_intervals,
                )
            )

    def _fail_outstanding_futures(self, exc: BaseException) -> None:
        """Fail every unresolved future after an exception or engine teardown."""
        for generation_future in self._generation_futures.values():
            if not generation_future.future.done():
                generation_future.future.set_exception(exc)
        self._generation_futures.clear()

        if self._pull_model_state_dict_future is not None:
            if not self._pull_model_state_dict_future.done():
                self._pull_model_state_dict_future.set_exception(exc)
            self._pull_model_state_dict_future = None
            self._model_state_dict_pull_request = None

    def _build_sampling_params(self, sampling: SamplingConfig) -> SamplingParams:
        """Translate a `SamplingConfig` into vLLM `SamplingParams` (n=1; seed from debug)."""
        return SamplingParams(
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            max_tokens=sampling.max_tokens,
            n=1,  # always expects a single sample per request. Caller can call N times.
            stop_token_ids=self._stop_token_ids or None,
            seed=self.config.debug.seed,
            logprobs=0,  # return only the sampled token's logprob (for the GRPO ratio)
            # Return each request's result once, when it is fully done, instead of streaming partial
            # outputs as tokens arrive.
            output_kind=RequestOutputKind.FINAL_ONLY,
        )

    @endpoint
    @sl.log_trace_span("pull_model_state_dict")
    async def pull_model_state_dict(self, version: int) -> None:
        """Queues a weight pull for `version` and blocks until the engine loop has finished pulling.

        NOTE: In-flight requests are NOT drained here — the endpoint never drains; a caller that wants
        an idle engine holds off new `generate` calls until the queue drains, then calls this.

        Args:
            version: Policy version to pull
        """
        # TODO: if an incoming request is received while another pull request is queued
        # we should drop the older request and pull the latest version instead

        # Starting requires asyncio, which isn't available in the sync __init__.
        # Start on first call; no-op after.
        await self._ensure_engine_loop()

        # Only rank 0 owns the queue + futures moving forward.
        if self._rank != 0:
            return

        # A placeholder future for the engine loop to resolve once the pull has been applied.
        pull_model_state_dict_future: asyncio.Future[
            int
        ] = asyncio.get_running_loop().create_future()

        # `_engine_loop_condition` wakes the engine loop, if asleep, when a pull is queued.
        async with self._engine_loop_condition:
            self._model_state_dict_pull_request = ModelStateDictPullRequest(
                version=version
            )
            self._pull_model_state_dict_future = pull_model_state_dict_future
            self._engine_loop_condition.notify()  # wakes the engine loop only if it is idle

        # Await outside the lock so other generate / pull calls can proceed meanwhile.
        await pull_model_state_dict_future

    @sl.log_trace_span("pull_model_state_dict_copy")
    async def _pull_model_state_dict(self, version: int) -> None:
        """ALL RANKS: collectively copy the latest weights from TorchStore, optionally drop the
        prefix cache (so no new request reuses an old-weight prefix), and bump the policy version.
        """
        # TODO: with >1 generator, trainer should probably use direct_rdma=False (CPU-staged, fanout-safe)
        # is_rdma_available() is a hardware probe, not a fanout signal.
        model_sd = self._get_model().model.state_dict()
        await ts.get_state_dict(
            "model_state_dict",
            user_state_dict=model_sd,
            strict=False,
            direct_rdma=is_rdma_available(),
        )
        self.policy_version = version
        if self.config.reset_prefix_cache_on_weight_sync:
            # TODO(async): under hot-swap, prefer per-token weight-version tracking (the episode
            # `version_intervals`) over a full cache drop.
            self._engine.reset_prefix_cache(
                reset_running_requests=self.config.reset_running_requests_on_weight_sync,
            )
        gc.collect()

        # Rank 0 holds the pull's future. Until this is resolved,
        # no new requests are admitted or processed.
        if self._rank == 0 and self._pull_model_state_dict_future is not None:
            self._pull_model_state_dict_future.set_result(version)
            self._pull_model_state_dict_future = None
            self._model_state_dict_pull_request = None

    @endpoint
    async def close(self) -> None:
        """Stop the engine loop, then release the vLLM engine.

        Rank 0 sets `_close_request` and notifies the engine_loop to quit the while-loop.
        Any futures the loop left unresolved are then failed, so awaiting callers get an
        exception instead of hanging.

        Engine teardown: with `external_launcher`, vLLM reuses the process group and actor
        lifetime that Monarch owns. Calling vLLM's internal `engine_core.shutdown()` can block
        while Monarch is also trying to stop the same actor mesh, so this endpoint only closes
        renderer-local resources and leaves process teardown to `ProcMesh.stop()`.
        """
        if self._rank == 0:
            async with self._engine_loop_condition:
                self._close_request = CloseRequest()
                self._engine_loop_condition.notify()  # wake the loop so it returns CLOSE

        # Let the engine loop process the shutdown.
        if self._engine_loop_task is not None:
            try:
                await self._engine_loop_task
            except Exception:
                logger.exception("engine loop raised during shutdown")
            self._engine_loop_task = None

        # The loop has stopped; fail any futures it left unresolved so awaiting callers get an
        # exception instead of hanging.
        self._fail_outstanding_futures(
            RuntimeError("generator closed before the request finished")
        )

        # Shut down engine parts
        if self._engine is not None:
            renderer = getattr(self._engine, "renderer", None)
            if renderer is not None:
                logger.info("Shutting down vLLM renderer")
                renderer.shutdown()
            self._engine = None


# ===================== helpers =====================


# ---- Engine-loop inbox: requests the controller submits (picklable; broadcast in LoopDecision). ----


@dataclass(kw_only=True, slots=True)
class GenerationRequest:
    """One queued `generate` call awaiting admission to the engine."""

    request_id: str
    prompt_token_ids: list[int]  # [prompt_tokens]
    sampling: SamplingConfig


@dataclass(kw_only=True, slots=True)
class ModelStateDictPullRequest:
    """A queued weight pull: the policy `version` to copy from TorchStore."""

    version: int


@dataclass(kw_only=True, slots=True)
class CloseRequest:
    """A queued shutdown signal (no payload); the engine loop returns CLOSE when it sees one."""


# ---- Rank-0 outbox: the future the loop resolves per generation. ----


@dataclass(kw_only=True, slots=True)
class GenerationFuture:
    """A generation request's future the loop resolves with its `Completion`. `metrics_prefix`
    namespaces the per-generation metrics built at completion (e.g. "generator" vs
    "validation_generator"). `version_intervals` records the policy version the request was
    admitted (sampled) at (see `Completion`)."""

    future: asyncio.Future[Completion]
    metrics_prefix: str
    version_intervals: list[tuple[int, int]] = field(default_factory=list)


class LoopAction(enum.Enum):
    """What the engine loop does each loop iteration (rank 0 decides; the choice is broadcast to all)."""

    STEP = "step"
    # run a burst of engine.step(); first admit any newly-queued requests.

    PULL_MODEL_STATE_DICT = "pull_model_state_dict"
    # pull the latest weights between bursts

    CLOSE = "close"
    # stop the loop


@dataclass(kw_only=True, slots=True)
class LoopDecision:
    """The per-iteration decision rank 0 broadcasts so every rank acts identically.
    This is broadcasted (pickled) in the engine_loop"""

    action: LoopAction

    requests: list[GenerationRequest] | None = None
    # requests to admit before a STEP burst (empty unless any queued)

    pull_version: int | None = None
    # set iff action is PULL_MODEL_STATE_DICT
