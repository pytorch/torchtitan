# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""vLLM-backed generator actor.

``generate_tokens`` is a batched endpoint: it accepts a list of
pre-tokenized prompts, submits them all to vLLM's engine in the same
order on every TP rank, drains via ``engine.step()`` until every
request finishes, returns the per-prompt :class:`GenerateOutput`\\ s.

Why batched: ``vLLM external_launcher`` TP requires every rank's
scheduler to see the same requests in the same order; a per-request
endpoint shape lets Monarch's per-rank message ordering diverge,
which breaks scheduler agreement and silently hangs NCCL. The
batched shape sends one ordered list to every rank.

Weight sync (``pull_model_state_dict``) runs between
``generate_tokens`` calls — endpoint dispatch on a single actor is
sequential, so no in-flight requests exist when it fires.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any

import torch
import torchstore as ts
from monarch.actor import Actor, current_rank, endpoint
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
from torchtitan.observability import structured_logger as sl
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools.logging import init_logger
from torchtitan.tools.utils import has_cuda_capability
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.config import AttentionConfig, CompilationConfig
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.attention.backends.registry import AttentionBackendEnum

logger = logging.getLogger(__name__)

__all__ = ["GenerateOutput", "SamplingConfig", "VLLMGenerator"]


# ---------------------------------------------------------------------------
# Wire types
# ---------------------------------------------------------------------------


@dataclass(kw_only=True, slots=True)
class GenerateOutput:
    """One response from ``VLLMGenerator.generate_tokens``.

    Shape legend:
        T_r: number of response tokens (= ``len(response_token_ids)``).
    """

    policy_version: int
    response_token_ids: list[int]  # [T_r]
    response_logprobs: list[float]  # [T_r]
    finish_reason: str | None = None  # "stop" | "length" | "abort"


# ---------------------------------------------------------------------------
# Sampling + CUDA-graph configs (shared with the trainer config tree)
# ---------------------------------------------------------------------------


@dataclass(kw_only=True, slots=True)
class VLLMCudagraphConfig:
    """CUDA graph capture settings for the vLLM inference engine.

    torch.compile is configured separately via ``CompileConfig`` at the
    ``RLTrainer`` level, shared by trainer and generator. Only CUDA
    graph capture, which is vLLM-specific, is controlled here.

    When enabled, vLLM captures the forward pass as a single CUDA graph
    ("full" mode). "piecewise" modes are intentionally excluded: they
    preclude CUDA graph capture despite being torch.compile-compatible
    post https://github.com/pytorch/torchtitan/pull/3142.
    """

    enable: bool = True

    def get_vllm_compilation_config(
        self, *, max_num_seqs: int
    ) -> CompilationConfig | None:
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
    """Sampling parameters passed to vLLM's :class:`SamplingParams`.

    ``n`` is the number of samples per prompt at the vLLM level. With
    GRPO + ``EnvBuilder.make_envs(group_size)`` the controller fans out
    siblings at the env level, so ``n=1`` here is the typical setting;
    raise only if you want vLLM-side sibling fanout for a single env.
    """

    n: int = 1
    temperature: float = 0.8
    top_p: float = 0.95
    max_tokens: int = 100

    stop_token_ids: list[int] = field(default_factory=list)
    """Renderer-derived stop tokens; the controller fills this per
    call. NOT a property of the actor; the actor never owns the
    renderer."""


# ---------------------------------------------------------------------------
# Per-request metrics emitted to the structured logger for trace correlation.
# ---------------------------------------------------------------------------


def _emit_request_metrics(output: RequestOutput, *, prefix: str) -> None:
    """Stream per-request vLLM timing into ``structured_logger``.

    Pulls from ``RequestOutput.metrics`` (``RequestStateStats``). With
    ``SamplingParams.n > 1`` vLLM stores one ``RequestStateStats`` per
    child; the parent exposes the LAST-finishing child's timeline.
    """
    values: dict[str, float] = {}
    if output.num_cached_tokens is not None:
        values[f"{prefix}/num_cached_tokens"] = float(output.num_cached_tokens)
    stats = output.metrics
    if stats is not None:
        values[f"{prefix}/queue_time_ms"] = (
            stats.scheduled_ts - stats.queued_ts
        ) * 1000
        if stats.num_generation_tokens > 0:
            values[f"{prefix}/time_to_first_token_ms"] = (
                stats.first_token_latency * 1000
            )
            values[f"{prefix}/prefill_time_ms"] = (
                stats.first_token_ts - stats.scheduled_ts
            ) * 1000
        if stats.num_generation_tokens > 1:
            decode_ms = (stats.last_token_ts - stats.first_token_ts) * 1000
            values[f"{prefix}/decode_time_ms"] = decode_ms
            values[f"{prefix}/inter_token_latency_ms"] = decode_ms / (
                stats.num_generation_tokens - 1
            )
    if values:
        sl.log_trace_scalar(values)


# ---------------------------------------------------------------------------
# VLLMGenerator
# ---------------------------------------------------------------------------


class VLLMGenerator(Actor, Configurable):
    """vLLM-backed generator actor.

    One :meth:`generate_tokens` call takes a list of prompts, submits
    them to vLLM's engine in order, drains until each finishes, and
    returns the outputs. The controller batches N concurrent rollouts'
    current-turn prompts into a single call so that every TP rank sees
    the requests in the same order (vLLM ``external_launcher`` TP
    relies on deterministic scheduling across ranks).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Generator actor configuration.

        TODO: expose an ``EngineConfig`` field to pass arbitrary
        configuration to the vLLM Engine.
        """

        parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
        """Parallelism configuration for the vLLM engine."""

        sampling: SamplingConfig = field(default_factory=SamplingConfig)
        """Default sampling parameters; per-call overrides allowed."""

        model_dtype: str = "bfloat16"
        """Data type for model weights — auto / float16 / bfloat16 / float32."""

        gpu_memory_limit: float = 0.9
        """Fraction of GPU memory to use for the vLLM engine (0.0 to 1.0)."""

        cudagraph: VLLMCudagraphConfig = field(default_factory=VLLMCudagraphConfig)
        """CUDA graph capture settings for the vLLM engine."""

        checkpoint: CheckpointManager.Config = field(
            default_factory=CheckpointManager.Config
        )
        """Controls whether the vLLM wrapper loads initial HF weights.
        In the RL loop this should stay disabled; weights arrive from
        TorchStore. For standalone inference, set ``enable=True`` +
        ``initial_load_in_hf=True``."""

        debug: DebugConfig = field(default_factory=DebugConfig)
        """Debug and determinism settings."""

        def __post_init__(self):
            # vLLM handles its own parallelism; we only apply TP via the
            # core parallelize function. Reject every other dim explicitly
            # so misconfiguration fails loudly at config time.
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
                    "evenly divisible by TP, which doesn't hold for inference."
                )
            if not p.disable_loss_parallel:
                raise ValueError(
                    "Generator requires disable_loss_parallel=True, "
                    f"got disable_loss_parallel={p.disable_loss_parallel}"
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
    ) -> None:
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
        # ``max_num_seqs`` controls vLLM's maximum batch dimension; also
        # bounds CUDA graph capture sizes.
        self._max_num_seqs = max_num_seqs

        registry_to_vllm(
            model_spec,
            parallelism=config.parallelism,
            compile_config=compile_config,
            checkpoint_config=config.checkpoint,
        )

        os.environ["VLLM_ATTENTION_BACKEND"] = "CUSTOM"
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"

        set_batch_invariance(config.debug.batch_invariant)
        self._set_determinism(config.debug)

        self.model_path = model_path

        engine_kwargs: dict[str, Any] = dict(
            model=model_path,
            trust_remote_code=True,
            # torchtitan custom config parser; builds PretrainedConfig
            # from ModelSpec rather than reading config.json from disk.
            config_format=TORCHTITAN_CONFIG_FORMAT,
            dtype=config.model_dtype,
            tensor_parallel_size=config.parallelism.tensor_parallel_degree,
            # Monarch already spawned TP workers via proc mesh.
            distributed_executor_backend="external_launcher",
            gpu_memory_utilization=config.gpu_memory_limit,
            enforce_eager=not config.cudagraph.enable,
            attention_config=AttentionConfig(backend=AttentionBackendEnum.CUSTOM),
            # RequestOutput.metrics enabled for per-request timing data.
            disable_log_stats=False,
            max_num_seqs=self._max_num_seqs,
        )
        # FA2 requires block_size to be a multiple of 256 (pre-Hopper).
        if not has_cuda_capability(9, 0):
            engine_kwargs["block_size"] = 256
        comp = config.cudagraph.get_vllm_compilation_config(
            max_num_seqs=self._max_num_seqs
        )
        if comp is not None:
            engine_kwargs["compilation_config"] = comp
        if config.debug.seed is not None:
            engine_kwargs["seed"] = config.debug.seed

        with sl.log_trace_span("vllm_init"):
            self._engine = LLMEngine.from_engine_args(EngineArgs(**engine_kwargs))
            logger.info("vLLM engine initialized")

        self.policy_version: int = 0
        # Monotonic request ID counter; vLLM uses these to track in-flight
        # requests. We never reuse IDs because vLLM keeps a finished-set
        # internally even after we drain.
        self._next_request_id: int = 0
        # ----- Admission/stepping split (TBR-style continuous batching) -----
        #
        # Background: ``LLMEngine.add_request`` + ``LLMEngine.step`` are not
        # safe to interleave from multiple concurrent coroutines (would
        # corrupt vLLM's batching state and desync TP-rank schedulers).
        # The naive fix — hold one lock for the full add+drain — serializes
        # whole calls and forces vLLM down to batch=1 between turns (see
        # round_018_opus_generator_batching_deepdive.md).
        #
        # TBR's fix (genai/msl/rl/backends/vllm/native_sampler): a single
        # daemon-equivalent driver owns the step loop. Callers admit
        # requests briefly under the same lock, then release it and await
        # per-request futures. Between every ``engine.step()`` iteration
        # the driver releases the lock so newly-arrived callers can admit;
        # the engine accumulates requests across many callers and keeps
        # the scheduler full across multi-turn rollouts.
        #
        # ``_engine_step_lock`` — held briefly by either:
        #   * the driver, for one ``engine.step()`` iteration, or
        #   * a caller, while it calls ``add_request`` for its prompts
        #     (a few microseconds per request).
        # ``_req_futures`` — request_id → asyncio.Future[RequestOutput],
        #   populated by callers under the lock, consumed by the driver.
        # ``_req_admit_versions`` — request_id → policy_version at admit
        #   time. Snapshotted under the lock so a weight swap mid-rollout
        #   stamps each turn with the correct version.
        # ``_engine_driver_task`` — created lazily on first endpoint call
        #   (so it binds to the actor's event loop).
        # ``_engine_shutdown`` — driver exit signal.
        self._engine_step_lock: asyncio.Lock = asyncio.Lock()
        self._req_futures: dict[str, asyncio.Future[RequestOutput]] = {}
        self._req_admit_versions: dict[str, int] = {}
        self._engine_driver_task: asyncio.Task | None = None
        self._engine_shutdown: asyncio.Event | None = None
        # ``_admit_gate`` — closed (cleared) while ``pull_model_state_dict``
        # is in progress so new ``generate_tokens`` callers wait for the
        # post-pull weights to publish instead of admitting against
        # to-be-stale weights and stretching the drain unboundedly under
        # a steady rollout stream. Set on construction (open by default).
        self._admit_gate: asyncio.Event | None = None
        # Kept for backward-compatibility with any external code that
        # still references ``_engine_lock``; pull_model_state_dict uses
        # the same lock to drain.
        self._engine_lock: asyncio.Lock = self._engine_step_lock

    @staticmethod
    def _set_determinism(debug: DebugConfig) -> None:
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
        """Access the model from the vLLM engine (returns a ``VLLMModelWrapper``)."""
        return self._engine.model_executor.driver_worker.get_model()

    # ------------------------------------------------------------------ endpoints

    @endpoint
    @sl.log_trace_span("generate_tokens")
    async def generate_tokens(
        self,
        prompts: list[list[int]],
        *,
        sampling_config: SamplingConfig | None = None,
    ) -> list[GenerateOutput]:
        """Submit a list of prompts, await individual completions, return
        outputs in submission order.

        Admission and stepping are decoupled: this endpoint briefly
        acquires ``_engine_step_lock`` to call ``engine.add_request`` for
        every prompt and stash per-request futures, then releases and
        awaits the futures. A single background driver task
        (:meth:`_engine_driver_loop`) owns the step loop and re-acquires
        the lock for one ``engine.step()`` at a time, releasing between
        iterations so other concurrent ``generate_tokens`` calls can
        admit. The engine accumulates requests across many concurrent
        callers and keeps the vLLM scheduler full across multi-turn
        rollouts (no more batch=1 stalls between turns).

        Every TP rank's actor runs this same code; the Monarch endpoint
        delivery preserves submission order across ranks (single sender
        per Monarch ValueMesh broadcast), so all ranks' drivers admit in
        the same order and the schedulers stay in lockstep.

        Args:
            prompts: ``[num_prompts][prompt_tokens]`` — already tokenized.
            sampling_config: per-call override of the actor's default
                :class:`SamplingConfig`. ``seed`` comes from
                ``config.debug.seed`` (not per-call).
        """
        if not prompts:
            return []

        # Wait for any pending weight pull to publish before admitting,
        # so this call doesn't sample against to-be-stale weights and
        # doesn't extend the post-backward drain. (``_ensure_engine_driver``
        # is invoked inside the lock below, after add_request, so it
        # can't race with the driver's exit decision.)
        if self._admit_gate is None:
            # First call ever — driver hasn't been created yet, gate is
            # implicitly open. Create both eagerly so the gate exists
            # before any pull tries to clear it.
            self._ensure_engine_driver()
        await self._admit_gate.wait()

        sc = sampling_config or self.config.sampling
        params = SamplingParams(
            n=sc.n,
            temperature=sc.temperature,
            top_p=sc.top_p,
            max_tokens=sc.max_tokens,
            stop_token_ids=list(sc.stop_token_ids) if sc.stop_token_ids else None,
            seed=self.config.debug.seed,
            logprobs=1,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )

        loop = asyncio.get_running_loop()
        futures: list[asyncio.Future[RequestOutput]] = []
        admit_version: int
        # Brief lock — just enough to call add_request for every prompt
        # and stash a future + version stamp for each. Releases before
        # the long wait so the driver can step + other callers can admit.
        async with self._engine_step_lock:
            admit_version = self.policy_version
            engine_inputs = self._engine.renderer.render_cmpl(
                [{"prompt_token_ids": p} for p in prompts]
            )
            first_id = self._next_request_id
            self._next_request_id += len(prompts)
            for i, engine_input in enumerate(engine_inputs):
                req_id = str(first_id + i)
                fut: asyncio.Future[RequestOutput] = loop.create_future()
                self._req_futures[req_id] = fut
                self._req_admit_versions[req_id] = admit_version
                futures.append(fut)
                self._engine.add_request(
                    request_id=req_id,
                    prompt=engine_input,
                    params=params,
                )

            # Restart the driver if it exited while we were waiting on
            # the gate. Called UNDER THE LOCK so the old driver (if
            # still running) cannot decide-to-exit-and-finish between
            # our check and our admission — both would race the lock,
            # and one wins. If we win, we admit + ensure; the next
            # driver iteration sees our requests. If the old driver
            # wins, it sees engine.has_unfinished_requests() == False
            # AT THE MOMENT IT CHECKED (before our admit), exits, and
            # we then take the lock, admit, and start a fresh driver.
            self._ensure_engine_driver()

            device = torch.cuda.current_device()
            logger.info(
                "generator admit batch=%d in_flight=%d peak_alloc=%.1fGB peak_reserved=%.1fGB",
                len(prompts),
                len(self._req_futures),
                torch.cuda.max_memory_allocated(device) / 1e9,
                torch.cuda.max_memory_reserved(device) / 1e9,
            )
            torch.cuda.reset_peak_memory_stats(device)

        # Lock released — driver drives step() concurrently and other
        # callers can pile in. Await completion of just our requests.
        with sl.log_trace_span("await_futures"):
            outputs = await asyncio.gather(*futures)
        return [
            self._to_generate_output(o, policy_version=admit_version) for o in outputs
        ]

    def _ensure_engine_driver(self) -> None:
        """Lazily create the engine driver task on first endpoint call.

        Constructed eagerly with the rest of state would bind to whichever
        loop happened to call ``__init__``; lazy construction binds it to
        the actor's running event loop instead.
        """
        if self._engine_driver_task is not None and not self._engine_driver_task.done():
            return
        self._engine_shutdown = asyncio.Event()
        self._admit_gate = asyncio.Event()
        self._admit_gate.set()  # open by default
        self._engine_driver_task = asyncio.create_task(
            self._engine_driver_loop(), name="engine_driver"
        )

    async def _engine_driver_loop(self) -> None:
        """Continuous-admission driver: step engine, complete futures,
        exit when idle (restart on next admit via :meth:`_ensure_engine_driver`).

        Loop body (TBR ``_server_loop`` analogue, sample_method.py:537;
        codex's `_drive_engine` matches):
        1. Acquire step lock; if engine idle and no pending futures,
           exit cleanly so next caller re-creates the task.
        2. Call ``engine.step()`` once under the lock.
        3. Release lock, resolve finished futures outside the lock so
           ``set_result`` callbacks don't block admissions waiting on
           the lock.
        4. ``await asyncio.sleep(0)`` — single yield bounds admission
           latency to one step (~50ms) for a caller racing for the lock.

        Multi-step bounding (TBR's ``max_steps_per_iteration=8`` in
        sampler_base.py:226) is unnecessary here because we already
        release the lock between every step.
        """
        engine = self._engine
        try:
            while not self._engine_shutdown.is_set():
                ready: list[tuple[asyncio.Future[RequestOutput], RequestOutput]] = []
                async with self._engine_step_lock:
                    if not engine.has_unfinished_requests():
                        # No outstanding work — exit cleanly. ``_ensure_engine_driver``
                        # will recreate the task on the next admit.
                        return
                    outputs = engine.step()
                    for o in outputs:
                        if not o.finished:
                            continue
                        fut = self._req_futures.pop(o.request_id, None)
                        self._req_admit_versions.pop(o.request_id, None)
                        if fut is not None:
                            ready.append((fut, o))

                # Resolve futures OUTSIDE the lock so set_result callbacks
                # don't block callers racing to admit on the next tick.
                for fut, output in ready:
                    if not fut.done():
                        fut.set_result(output)

                await asyncio.sleep(0)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.exception("engine driver loop crashed")
            # Cascade the failure to all in-flight callers so they don't
            # hang forever on futures that will never resolve.
            crash_error = RuntimeError("engine driver crashed")
            crash_error.__cause__ = exc
            for fut in self._req_futures.values():
                if not fut.done():
                    fut.set_exception(crash_error)
            self._req_futures.clear()
            self._req_admit_versions.clear()
            raise

    @endpoint
    @sl.log_trace_span("pull_model_state_dict")
    async def pull_model_state_dict(self, version: int) -> None:
        """Drain in-flight requests, pull new weights, reset prefix cache.

        Tokens generated mid-pull would be a mix of old + new weights
        (KV cache from the old, future logits from the new), so we MUST
        drain before swapping. The driver continues stepping in the
        background as we wait; when ``_req_futures`` empties the engine
        has no in-flight work and we can take the step lock for the
        pull itself.

        Lock acquisition during the pull also prevents new callers from
        admitting requests against to-be-stale weights — they'll wait on
        the lock and admit once the new version is published.
        """
        from monarch.rdma import is_rdma_available

        from torchtitan.experiments.rl.actors.trainer import _dedup_tied_tensors

        self._ensure_engine_driver()
        # Close the admit gate so any new ``generate_tokens`` callers
        # park before they call ``add_request``. Without this, a steady
        # rollout stream during the drain would keep adding new requests
        # and the drain loop would never see ``_req_futures`` empty.
        gate_was_open = False
        if self._admit_gate is not None:
            gate_was_open = self._admit_gate.is_set()
            self._admit_gate.clear()
        try:
            # Wait for the driver to drain in-flight rollouts. The driver
            # task is actively stepping in the background, so this just
            # yields between checks until everything completes.
            while self._req_futures:
                await asyncio.sleep(0)

            async with self._engine_step_lock:
                model_sd = self._get_model().model.state_dict()
                # See trainer push for the rationale: the trainer dedups
                # tied tensors before publishing; do the same here so the
                # transfer plan on the receiver matches.
                await ts.get_state_dict(
                    "model_state_dict",
                    user_state_dict=_dedup_tied_tensors(model_sd),
                    strict=False,
                    direct_rdma=is_rdma_available(),
                )
                self._engine.reset_prefix_cache()
                self.policy_version = version
        finally:
            # Re-open the admit gate so parked callers can proceed.
            if gate_was_open and self._admit_gate is not None:
                self._admit_gate.set()

    @endpoint
    async def close(self) -> None:
        """Tear down vLLM in the same order as ``AsyncLLM``.

        1. Cancel the engine driver task (stops the step loop cleanly).
        2. ``renderer.shutdown()`` — close thread pools / multimodal cache.
        3. ``engine_core.shutdown()`` — stop the model worker + scheduler.
        4. ``cleanup_dist_env_and_memory()`` — destroy NCCL groups.
        """
        if self._engine_shutdown is not None:
            self._engine_shutdown.set()
        if self._engine_driver_task is not None and not self._engine_driver_task.done():
            self._engine_driver_task.cancel()
            try:
                await self._engine_driver_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._engine is not None:
            renderer = getattr(self._engine, "renderer", None)
            try:
                if renderer is not None:
                    renderer.shutdown()
            finally:
                self._engine.engine_core.shutdown()
        cleanup_dist_env_and_memory()

    # ------------------------------------------------------------------ internal

    def _to_generate_output(
        self, output: RequestOutput, *, policy_version: int
    ) -> GenerateOutput:
        """Build a :class:`GenerateOutput` from one finished ``RequestOutput``.

        ``policy_version`` is the weight version snapshotted at admission
        under ``_engine_lock`` — NOT ``self.policy_version`` at output time,
        which would mis-stamp every turn rolled out across a weight swap.

        Assumes ``SamplingParams.n == 1`` (the controller fans GRPO siblings
        out at the env-builder layer). With ``n > 1`` we return the first
        sample only.
        """
        # ``logprobs=1`` was passed; each entry is ``{chosen_id: Logprob(...)}``.
        sample = output.outputs[0]
        per_token_lp = [list(d.values())[0].logprob for d in sample.logprobs]
        _emit_request_metrics(output, prefix="generator")
        return GenerateOutput(
            policy_version=policy_version,
            response_token_ids=list(sample.token_ids),
            response_logprobs=per_token_lp,
            finish_reason=sample.finish_reason,
        )
