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
        # Serializes concurrent ``generate_tokens`` invocations on the
        # actor's event loop. Monarch's Direct dispatch lets multiple
        # endpoint coroutines run concurrently; without this lock,
        # two ``generate_tokens`` calls interleave ``engine.step``
        # invocations, which corrupts vLLM's continuous-batching state
        # and hangs NCCL on a subsequent TP collective.
        # Constructed eagerly: asyncio.Lock() binds to the running loop
        # on first ``async with``, so no event loop is required here.
        self._engine_lock: asyncio.Lock = asyncio.Lock()

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
        """Submit a list of prompts, drive ``engine.step()`` to completion,
        return outputs in submission order.

        The controller serializes N concurrent rollouts into one batched
        call so that every TP rank's engine sees the same ordered prompt
        list and the schedulers stay in lockstep (vLLM ``external_launcher``
        TP requires deterministic scheduling across ranks). The
        per-request submit-queue shape was incorrect for this topology
        — see round 7 §4.

        Args:
            prompts: ``[num_prompts][prompt_tokens]`` — already tokenized.
            sampling_config: per-call override of the actor's default
                :class:`SamplingConfig`. ``seed`` comes from
                ``config.debug.seed`` (not per-call).
        """
        if not prompts:
            return []

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

        # Serialize: concurrent generate_tokens invocations must not
        # interleave engine.add_request / engine.step pairs (would
        # corrupt vLLM's continuous-batching state across TP ranks).
        # ``pull_model_state_dict`` takes the same lock, so weights
        # cannot swap mid-batch — every output in this call belongs to
        # ``admit_version`` snapshotted under the lock.
        async with self._engine_lock:
            admit_version = self.policy_version
            engine_inputs = self._engine.renderer.render_cmpl(
                [{"prompt_token_ids": p} for p in prompts]
            )
            first_id = self._next_request_id
            self._next_request_id += len(prompts)
            for i, engine_input in enumerate(engine_inputs):
                self._engine.add_request(
                    request_id=str(first_id + i),
                    prompt=engine_input,
                    params=params,
                )

            # Drain the engine in-process; yield between steps so the
            # actor event loop stays responsive.
            finished: dict[str, RequestOutput] = {}
            target = {str(first_id + i) for i in range(len(prompts))}
            with sl.log_trace_span("engine_steps"):
                while target - finished.keys():
                    outputs = self._engine.step()
                    await asyncio.sleep(0)
                    for o in outputs:
                        if o.finished and o.request_id in target:
                            finished[o.request_id] = o

            # Per-batch memory surface to console — proper Metric() wiring
            # through MetricsProcessor is a follow-up. vLLM v1's LLMEngine
            # doesn't expose ``.scheduler.running`` directly, so we report
            # batch size + peak memory only.
            device = torch.cuda.current_device()
            logger.info(
                "generator batch=%d peak_alloc=%.1fGB peak_reserved=%.1fGB",
                len(prompts),
                torch.cuda.max_memory_allocated(device) / 1e9,
                torch.cuda.max_memory_reserved(device) / 1e9,
            )
            torch.cuda.reset_peak_memory_stats(device)

            return [
                self._to_generate_output(
                    finished[str(first_id + i)], policy_version=admit_version
                )
                for i in range(len(prompts))
            ]

    @endpoint
    @sl.log_trace_span("pull_model_state_dict")
    async def pull_model_state_dict(self, version: int) -> None:
        """Pull new weights from TorchStore, reset prefix cache, bump version.

        Acquires ``_engine_lock`` to serialize with ``generate_tokens``:
        Monarch's Direct dispatch lets endpoint coroutines run
        concurrently, so without this lock a pull could overlap with
        an in-flight batch and corrupt the engine. The lock is the
        single serialization point; the rest of the actor stays
        thread/coroutine-unsafe by design.
        """
        from monarch.rdma import is_rdma_available

        from torchtitan.experiments.rl.actors.trainer import _dedup_tied_tensors

        async with self._engine_lock:
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

    @endpoint
    async def close(self) -> None:
        """Tear down vLLM in the same order as ``AsyncLLM``.

        1. ``renderer.shutdown()`` — close thread pools / multimodal cache.
        2. ``engine_core.shutdown()`` — stop the model worker + scheduler.
        3. ``cleanup_dist_env_and_memory()`` — destroy NCCL groups.
        """
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
