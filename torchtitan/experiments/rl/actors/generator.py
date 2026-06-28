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
from typing import Literal

import cloudpickle
import torch
import torch.distributed as dist
import torchstore as ts
from monarch.actor import Actor, Channel, current_rank, endpoint, Port, PortReceiver
from torch.distributed.tensor import DTensor
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.config import CompileConfig, Configurable, DebugConfig, OverrideConfig
from torchtitan.distributed.utils import get_spmd_backend, set_batch_invariance
from torchtitan.experiments.rl.batch_invariance import (
    force_logprobs_fn_for_batch_invariance,
    patch_bmm_for_batch_invariance,
)
from torchtitan.experiments.rl.models.vllm_registry import (
    InferenceParallelismConfig,
    register_to_vllm,
    TORCHTITAN_CONFIG_FORMAT,
)
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.routing.intra_generator_router import (
    IntraGeneratorRouter,
)
from torchtitan.experiments.rl.types import Completion
from torchtitan.models.common.attention import (
    FlexAttention,
    FusedQKVLinear,
    VarlenAttention,
)
from torchtitan.observability import structured_logger as sl
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.protocols.sharding import resolve_placements, SpmdLayout
from torchtitan.tools.logging import init_logger
from torchtitan.tools.utils import has_cuda_capability
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.config import AttentionConfig, CompilationConfig, ParallelConfig
from vllm.config.compilation import CompilationMode
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.attention.backends.registry import AttentionBackendEnum

logger = logging.getLogger(__name__)

# TODO(async-rl): this file is large. Split a backend-agnostic BaseGenerator.


def _fqn_to_spmd_layout(model: torch.nn.Module) -> dict[str, SpmdLayout]:
    """Return state-dict FQN -> SPMD layout for generator weight pulls."""
    layouts: dict[str, SpmdLayout] = {}

    for module_fqn, module in model.named_modules():
        sharding_config = getattr(module, "_sharding_config", None)
        if sharding_config is not None:
            for state_name, layout in sharding_config.state_shardings.items():
                fqn = f"{module_fqn}.{state_name}" if module_fqn else state_name
                layouts[fqn] = layout

        if isinstance(module, FusedQKVLinear):
            # FusedQKVLinear exposes split wq/wk/wv state-dict keys while the
            # sharding layout lives on the fused wqkv parameter.
            wqkv_sharding_config = getattr(module.wqkv, "_sharding_config", None)
            if wqkv_sharding_config is None:
                continue
            for state_name, layout in wqkv_sharding_config.state_shardings.items():
                for proj_name in ("wq", "wk", "wv"):
                    layouts[f"{module_fqn}.{proj_name}.{state_name}"] = layout

    return layouts


def _is_fused_qkv_state_key(name: str) -> bool:
    parts = name.split(".")
    return "attention" in parts and any(proj in parts for proj in ("wq", "wk", "wv"))


def _tensor_debug_sample(tensor: torch.Tensor) -> str:
    if tensor.numel() == 0:
        return "empty"
    flat = tensor.detach().flatten()
    return f"first={flat[0].item()} last={flat[-1].item()}"


@dataclass(kw_only=True, slots=True)
class _RequestMetricsInputs:
    """Raw inputs needed to build a request's vLLM metrics. Used to pass
    metric related information when fan-in from DPs to rank 0.
    """

    num_cached_tokens: int | None
    has_stats: bool
    queued_ts: float = 0.0
    scheduled_ts: float = 0.0
    first_token_ts: float = 0.0
    last_token_ts: float = 0.0
    first_token_latency: float = 0.0
    num_generation_tokens: int = 0


def _extract_request_metrics_inputs(
    request_output: RequestOutput,
) -> _RequestMetricsInputs:
    """Pull the raw metric inputs off a finished ``RequestOutput``."""
    stats = request_output.metrics
    if stats is None:
        return _RequestMetricsInputs(
            num_cached_tokens=request_output.num_cached_tokens, has_stats=False
        )
    return _RequestMetricsInputs(
        num_cached_tokens=request_output.num_cached_tokens,
        has_stats=True,
        queued_ts=stats.queued_ts,
        scheduled_ts=stats.scheduled_ts,
        first_token_ts=stats.first_token_ts,
        last_token_ts=stats.last_token_ts,
        first_token_latency=stats.first_token_latency,
        num_generation_tokens=stats.num_generation_tokens,
    )


def _prepare_generation_request_metrics(
    inputs: _RequestMetricsInputs, *, prefix: str
) -> list[m.Metric]:
    """Prepare vLLM per-request metrics from the raw inputs.

    For `add_request` call, vLLM returns a RequestOutput carrying
    a single `RequestStateStats` (captured into `_RequestMetricsInputs`).

    Caveat under `SamplingParams.n > 1`: vLLM stores one `RequestStateStats`
    per child request; the parent output exposes the **last-finishing**
    child's timeline. `arrival_time` is shared across siblings, but
    [`queued_ts`, `scheduled_ts`, `first_token_ts`, `last_token_ts`,
    `num_generation_tokens`] describe one specific child - not an aggregate,
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
    if inputs.num_cached_tokens is not None:
        metric_values[f"{prefix}/num_cached_tokens"] = inputs.num_cached_tokens

    if inputs.has_stats:
        metric_values[f"{prefix}/queue_time_ms"] = (
            inputs.scheduled_ts - inputs.queued_ts
        ) * 1000

        if inputs.num_generation_tokens > 0:
            metric_values[f"{prefix}/time_to_first_token_ms"] = (
                inputs.first_token_latency * 1000
            )
            metric_values[f"{prefix}/prefill_time_ms"] = (
                inputs.first_token_ts - inputs.scheduled_ts
            ) * 1000

        if inputs.num_generation_tokens > 1:
            first_to_last_token_ms = (
                inputs.last_token_ts - inputs.first_token_ts
            ) * 1000
            metric_values[f"{prefix}/decode_time_ms"] = first_to_last_token_ms
            metric_values[
                f"{prefix}/inter_token_latency_ms"
            ] = first_to_last_token_ms / (inputs.num_generation_tokens - 1)

    # Emit each value with both Mean and Max aggregators.
    return [
        metric
        for key, value in metric_values.items()
        for metric in (m.Metric(key, m.Mean(value)), m.Metric(key, m.Max(value)))
    ]


# vLLM's default max_num_batched_tokens (vllm's per-step budget:
# prefill + decode tokens summed over the batch). Used as the cudagraph capture
# cap for "FULL" / "FULL_AND_PIECEWISE" (which graph prefill / mixed batches, so
# capture sizes must reach the per-step budget or those batches fall back to
# eager) when ``Config.max_num_batched_tokens`` is unset; when that field is set,
# its value is used instead (and also drives the vLLM engine).
_DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048


@dataclass(kw_only=True, slots=True)
class VLLMCudagraphConfig:
    """CUDA graph capture settings for the vLLM inference engine.

    torch.compile is configured separately via ``CompileConfig`` at the
    ``Controller`` level, shared by both trainer and generator.  Only CUDA
    graph capture, which is vLLM-specific, is controlled here.

    ``mode`` selects which vLLM cudagraph mode to capture; see that field and
    ``get_vllm_compilation_config`` for the per-mode trade-offs. The default,
    ``FULL_DECODE_ONLY``, is the only mode that is both cheap (no inductor
    compile) and correct with our varlen/FA3 attention backend.
    """

    enable: bool = True
    """Whether to enable CUDA graph capture."""

    mode: Literal["FULL_DECODE_ONLY", "FULL_AND_PIECEWISE", "FULL"] = "FULL_DECODE_ONLY"
    """Which vLLM cudagraph mode to capture (when ``enable``):

    - ``"FULL_DECODE_ONLY"`` (default): graph pure-decode batches; prefill / mixed
      batches run eager. Cheap (no inductor compile) and correct with our
      varlen/FA3 attention backend (#3668).
    - ``"FULL_AND_PIECEWISE"``: FULL graph for pure single-token decode (whole
      forward incl. attention -- safe because decode has a fixed query_len==1
      layout) AND breakable PIECEWISE for prefill / mixed batches (attention runs
      eager at a stream-capture break). Best coverage: the common decode path
      gets a full graph while only mixed batches pay the eager-break cost.
      Requires ``VLLM_USE_BREAKABLE_CUDAGRAPH=1``.
    - ``"FULL"``: graph the whole forward, prefill included, attention captured
      too. Only valid with the flex attention backend, which survives FULL
      capture of mixed prefill+decode batches
    """

    capture_sizes: list[int] | None = None
    """Explicit cudagraph capture batch sizes. When ``None`` (default), sizes are
    auto-derived: powers of 2 up to the cap, plus ``max_num_seqs`` and the cap as
    exact sizes. When set, exactly these sizes are captured (deduped and sorted)."""

    # TODO: Validate CUDA graph capture with MoE / Expert Parallelism.
    # MoE routing produces dynamic shapes that may conflict with full
    # CUDA graph capture despite being torch.compile-compatible
    # post https://github.com/pytorch/torchtitan/pull/3142

    # TODO: Explore applying CUDA graph capture on the torchtitan trainer
    # side as well (not just the vLLM generator).
    # https://github.com/pytorch/torchtitan/issues/3175

    def get_vllm_compilation_config(
        self, *, max_num_seqs: int, max_num_batched_tokens: int | None = None
    ) -> CompilationConfig | None:
        """Build a vLLM ``CompilationConfig`` for ``mode``, or return ``None``
        when CUDA graphs are disabled.

        When ``capture_sizes`` is set, those exact sizes are captured. Otherwise
        sizes are auto-derived: powers of 2 up to the cap, plus ``max_num_seqs`` and
        the cap itself as exact sizes so the largest capture size is always the cap
        (even when it is not a power of 2). The cap is ``max_num_seqs`` for
        ``FULL_DECODE_ONLY`` (decode batch == num_seqs). ``FULL`` and
        ``FULL_AND_PIECEWISE`` also graph prefill, whose per-step token count is
        bounded by ``max_num_batched_tokens`` (the configured value, else
        ``_DEFAULT_MAX_NUM_BATCHED_TOKENS``), so the cap extends to it
        -- otherwise prefill chunks larger than the cap fall back to eager.

        All modes capture with ``mode=CompilationMode.NONE`` (no inductor compile).
        ``FULL_AND_PIECEWISE`` runs attention eager via vLLM's BREAKABLE
        cudagraph, which requires ``VLLM_USE_BREAKABLE_CUDAGRAPH=1`` (vLLM itself
        also forces ``mode=NONE`` when that env is set) (#3709).
        """
        if not self.enable:
            return None
        if max_num_seqs <= 0:
            raise ValueError(f"max_num_seqs must be positive, got {max_num_seqs}")
        if max_num_batched_tokens is not None:
            _max_cudagraph_capture_size = max_num_batched_tokens
        else:
            _max_cudagraph_capture_size = _DEFAULT_MAX_NUM_BATCHED_TOKENS
        cap = max_num_seqs
        if self.mode in ("FULL", "FULL_AND_PIECEWISE"):
            cap = max(cap, _max_cudagraph_capture_size)
        if self.capture_sizes is not None:
            if not self.capture_sizes or any(s <= 0 for s in self.capture_sizes):
                raise ValueError(
                    "cudagraph.capture_sizes must be a non-empty list of positive "
                    f"ints, got {self.capture_sizes}"
                )
            sizes = sorted(set(self.capture_sizes))
        else:
            sizes = [1 << i for i in range(int(math.log2(cap)) + 1)]
            # Always include max_num_seqs (decode batch) and the cap (largest
            # prefill chunk) as exact sizes so the largest capture size is the cap
            # even when it is not a power of 2
            if max_num_seqs not in sizes:
                sizes.append(max_num_seqs)
            if cap not in sizes:
                sizes.append(cap)
            sizes = sorted(sizes)

        return CompilationConfig(
            cudagraph_mode=self.mode,
            mode=CompilationMode.NONE,
            cudagraph_capture_sizes=sizes,
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

    seed: int | None = None
    """Per-request RNG seed. The rollouter offsets this per sample so a group's
    n=1 requests stay diverse while remaining reproducible (None = nondeterministic)."""

    stop_token_ids: list[int] | None = None
    """Renderer role-boundary stop tokens; filled by the controller."""


class RequestDispatcher:
    """Owns the generator's DP/TP request dispatch, hiding the rank layout behind
    a small interface so ``VLLMGenerator`` does not deal with it directly.

    Every rank holds one dispatcher; methods act according to the rank's role:
    - Rank 0 is the coordinator (and DP0's tp_rank=0): it holds the generation
      futures, routes requests, opens the fan-in port, and resolves every
      completion -- its own replica's locally, peers' via the drain task. State
      and methods only ever used on rank 0 are prefixed ``rank0_``.
    - Other DP's tp_rank=0 build finished completions and fan them in
      to rank 0 over the port.
    - tp_rank!=0 hold no outputs; their dispatcher only carries the layout.

    Supported rank layout:

        global_rank = dp_rank * tp_degree + tp_rank

    EP reuses the same global-rank -> (dp, tp) mapping, so it needs no special
    handling here.

    Data flow:

    Take DP=2, TP=2 for example. Only rank 0 holds futures and talks to the
    controller, so completions produced by any other DP replica must be sent
    back ("fanned in") to rank 0:

        controller --generate--> rank 0  (registers a future)
                                   |
            rank0_route(): pick a DP rank for the queued requests
                                   |   (broadcast in the LoopDecision, elsewhere)
              +--------------------+--------------------+
              v                                         v
        DP0's tp_rank=0 (i.e. rank 0)               DP1's tp_rank=0 (i.e. rank 2)
          engine.step()                             engine.step()
          build (completion, metrics_inputs)        build (completion, metrics_inputs)
          resolve own futures locally   <--port--   send completions to rank 0
                                                    (tp_rank!=0: hold no outputs)
        rank 0 background drain task: recv from port -> resolve those futures
    """

    def __init__(
        self,
        *,
        rank: int,
        parallelism: InferenceParallelismConfig,
        broadcast_group: dist.ProcessGroup,
        vllm_parallel_config: ParallelConfig,
        intra_generator_router: IntraGeneratorRouter.Config,
    ):
        self._rank = rank
        # Only DP and TP are supported, so ``tp_degree`` ranks make up one DP
        # replica. EP does not change the global-rank -> (dp, tp) mapping, so it
        # needs no handling here. TODO: revisit if PP/CP are ever added.
        self._dp_degree = parallelism.data_parallel_degree
        tp_degree = parallelism.tensor_parallel_degree
        # Which DP replica this rank belongs to (== vLLM's data_parallel_rank).
        self._dp_rank = rank // tp_degree
        # Rank within the DP replica
        self._tp_rank = rank % tp_degree
        # Reused for the one-time result-port broadcast (see ``setup``).
        self._broadcast_group = broadcast_group

        # Confirm our derived layout matches what vLLM computed independently.
        assert vllm_parallel_config.data_parallel_size == self._dp_degree, (
            f"DP layout mismatch on rank {self._rank}: our dp_size "
            f"({self._dp_degree}) != vLLM data_parallel_size "
            f"({vllm_parallel_config.data_parallel_size})"
        )
        assert vllm_parallel_config.data_parallel_rank == self._dp_rank, (
            f"DP layout mismatch on rank {self._rank}: our dp_rank "
            f"({self._dp_rank}) != vLLM data_parallel_rank "
            f"({vllm_parallel_config.data_parallel_rank})"
        )

        # RANK-0 OUTBOX: futures the engine loop resolves so the awaiting endpoint
        # returns. Only rank 0 ever populates this.
        self._rank0_generation_futures: dict[str, GenerationFuture] = {}

        # --- Result fan-in (only when DP>1) ---
        # rank 0 opens the channel, keeps the receiving end, and its drain task
        # resolves whatever peer tp_rank=0 send. ``_result_port`` is the sending
        # end, held only by peer tp_rank=0; it stays None on rank 0 (resolves
        # locally) and on tp_rank!=0 (no outputs). All None for DP=1 since there
        # is no peer DP to send from.
        self._result_port: Port | None = None
        self._rank0_result_receiver: PortReceiver | None = None
        self._rank0_drain_task: asyncio.Task | None = None

        # --- DP routing ---
        # RANK-0 DP routing: pick a DP rank per request, reserving its load until
        # the completion resolves.
        self._rank0_dp_router: IntraGeneratorRouter | None = (
            intra_generator_router.build(dp_degree=self._dp_degree)
            if self._rank == 0 and self._dp_degree > 1
            else None
        )

    def rank0_register_future(
        self, request_id: str, metrics_prefix: str
    ) -> asyncio.Future[Completion]:
        """RANK 0: register a future for ``request_id`` and return it to await."""
        if request_id in self._rank0_generation_futures:
            raise ValueError(f"request_id {request_id!r} is already in flight")
        future: asyncio.Future[Completion] = asyncio.get_running_loop().create_future()
        self._rank0_generation_futures[request_id] = GenerationFuture(
            future=future, metrics_prefix=metrics_prefix
        )
        return future

    def rank0_has_pending_futures(self) -> bool:
        """RANK 0: whether any request is still in flight (future unresolved).

        A future stays registered until its completion comes back, so this stays
        True while any peer DP rank is still running.
        """
        return bool(self._rank0_generation_futures)

    def rank0_route(
        self, requests: list[GenerationRequest]
    ) -> list[list[GenerationRequest]]:
        """RANK 0: pick which DP rank serves each queued request.

        Returns a fixed-length (``dp_degree``) list; index == DP rank. Each rank
        later admits only its own slice. With a single DP rank, everything goes
        to DP rank 0; otherwise ``IntraGeneratorRouter`` reserves a DP rank per
        request and that reservation is released when the request's completion
        resolves.
        """
        requests_per_dp_rank: list[list[GenerationRequest]] = [
            [] for _ in range(self._dp_degree)
        ]
        for request in requests:
            if self._rank0_dp_router is None:
                dp_rank = 0
            else:
                # Pick a DP rank for this request, and increment this DP rank's
                # load by 1 (i.e. measured by request count).
                dp_rank = self._rank0_dp_router.reserve(
                    request.request_id,
                    routing_session_id=request.routing_session_id,
                )
            requests_per_dp_rank[dp_rank].append(request)

        return requests_per_dp_rank

    def rank0_stamp_min_policy_version(
        self,
        requests_per_dp_rank: list[list[GenerationRequest]],
        policy_version: int,
    ) -> None:
        """RANK 0: stamp the admitted (sampling) version on every future in this STEP
        decision, across all DP ranks. Rank 0 owns the futures regardless of which DP
        rank serves the request, so it stamps them all here."""
        for dp_requests in requests_per_dp_rank:
            for request in dp_requests:
                self._rank0_generation_futures[
                    request.request_id
                ].min_policy_version = policy_version

    def setup(self) -> None:
        """One-time setup before the engine loop starts (DP>1): distribute rank 0's
        result-fan-in port and start rank 0's drain task.

        All ranks call this so the broadcast over the all-ranks ``_broadcast_group``
        completes; only TP rank 0 keep the port.
        """
        if self._dp_degree == 1:
            return

        # rank 0 opens the channel and keeps the receiving end; it broadcasts only
        # the sending port to the peers.
        if self._rank == 0:
            port, self._rank0_result_receiver = Channel.open()
            # Monarch Port objects need cloudpickle, so we cloudpickle it into bytes
            # first. Otherwise, broadcast_object_list will attempt to pickle the
            # port object with stdlib pickle and result in error.
            container = [cloudpickle.dumps(port)]
        else:
            container = [None]
        dist.broadcast_object_list(
            container, src=0, group=self._broadcast_group, device=torch.device("cpu")
        )
        assert container[0] is not None
        # Only peer TP rank 0s send completions to global rank 0, so only they
        # keep the port. Global rank 0 resolves locally and tp_rank!=0 produce
        # no outputs.
        if self._rank != 0 and self._tp_rank == 0:
            self._result_port = cloudpickle.loads(container[0])
        if self._rank == 0:
            self._rank0_drain_task = asyncio.create_task(self._rank0_drain_results())

    def process_finished_requests(
        self, request_outputs: list[RequestOutput], policy_version: int
    ) -> None:
        """TP rank 0s send finished completions to global rank 0 after each
        ``engine.step()``:
          - Global Rank 0 resolves its own DP replica's completions locally;
          - every other TP rank 0 sends them over the port to global rank 0's drain
            task.
          - Other ranks hold no finished outputs and do nothing.
        """
        if self._tp_rank != 0:
            return

        completions = self._build_completions(request_outputs, policy_version)
        if self._rank == 0:
            self._rank0_resolve_futures(completions)
        elif completions:
            self._result_port.send(completions)

    def _build_completions(
        self, request_outputs: list[RequestOutput], policy_version: int
    ) -> list[tuple[str, Completion, _RequestMetricsInputs]]:
        """Turn finished ``RequestOutput``s into ``(request_id, Completion, metrics_inputs)``."""
        completions: list[tuple[str, Completion, _RequestMetricsInputs]] = []
        for request_output in request_outputs:
            # We enforce n=1 in sampling params -> exactly one CompletionOutput per finished request
            # Here we just sanity check it (a single engine.step may still finish several requests).
            if len(request_output.outputs) != 1:
                raise ValueError(
                    f"expected n=1 (one sample per request), got "
                    f"{len(request_output.outputs)} for {request_output.request_id}"
                )

            # get logprobs
            completion_output = request_output.outputs[0]
            token_logprobs = [
                next(iter(logprob_dict.values())).logprob
                for logprob_dict in completion_output.logprobs
            ]

            completions.append(
                (
                    request_output.request_id,
                    Completion(
                        # NOTE: min_policy_version is a PLACEHOLDER here, set equal to max (the finish
                        # version). The serving rank has no access to the future that holds the true
                        # admitted version, so rank 0 REPLACES this with the real value in
                        # _rank0_resolve_futures. min == max here ONLY until that replacement.
                        min_policy_version=policy_version,
                        max_policy_version=policy_version,
                        request_id=request_output.request_id,
                        token_ids=list(completion_output.token_ids),
                        token_logprobs=token_logprobs,
                        finish_reason=completion_output.finish_reason,
                    ),
                    _extract_request_metrics_inputs(request_output),
                )
            )
        return completions

    def _rank0_resolve_futures(
        self, completions: list[tuple[str, Completion, _RequestMetricsInputs]]
    ) -> None:
        """RANK 0: build each completion's metrics (the only place that knows the
        request's ``metrics_prefix``), then resolve its future.

        TODO: metrics are built in two phases -- a DP-leader produces the raw
        ``_RequestMetricsInputs`` alongside the ``Completion``, and rank 0
        finalizes ``completion.metrics`` in place here, where it has the
        ``inflight_requests_at_completion`` count. Consider unifying into a
        single build step once that count can travel with (or be derived
        without) the rank-0 future bookkeeping.
        """
        for request_id, completion, metrics_inputs in completions:
            # in flight when this one finished (includes itself; counted before the pop)
            inflight_requests_at_completion = float(len(self._rank0_generation_futures))
            generation_future = self._rank0_generation_futures.pop(request_id)

            # Replace the placeholder min (the builder set min == max) with the true admitted
            # version stamped on the future at admission.
            completion.min_policy_version = generation_future.min_policy_version
            metrics_prefix = generation_future.metrics_prefix

            metrics = _prepare_generation_request_metrics(
                metrics_inputs, prefix=metrics_prefix
            )
            for metric_type in [m.Max, m.Mean]:
                metrics.append(
                    m.Metric(
                        f"{metrics_prefix}/inflight_requests_at_completion",
                        metric_type(inflight_requests_at_completion),
                    )
                )
            completion.metrics = metrics

            generation_future.future.set_result(completion)
            # Free the request's reserved load on its DP rank so load-aware
            # routing sees the accurate loads on DPs.
            if self._rank0_dp_router is not None:
                self._rank0_dp_router.release(request_id)

    async def _rank0_drain_results(self) -> None:
        """RANK 0 background task which receives and resolves completions pushed
        by peer TP rank 0s.
        """
        while True:
            completions = await self._rank0_result_receiver.recv()
            self._rank0_resolve_futures(completions)

    def fail_generation_futures(self, exc: BaseException) -> None:
        """RANK 0: fail every unresolved generation future after an exception or
        teardown (no-op elsewhere, where the map is empty)."""
        for generation_future in self._rank0_generation_futures.values():
            if not generation_future.future.done():
                generation_future.future.set_exception(exc)
        self._rank0_generation_futures.clear()

    async def shutdown(self) -> None:
        """Stop rank 0's drain task, if any (no-op elsewhere)."""
        if self._rank0_drain_task is not None:
            self._rank0_drain_task.cancel()
            try:
                await self._rank0_drain_task
            except (asyncio.CancelledError, Exception):
                pass
            self._rank0_drain_task = None


class VLLMGenerator(Actor, Configurable):
    """vLLM engine to drive concurrent `generate` calls through one SPMD engine loop.

    The controller fires independent calls (`generate`, `pull_model_state_dict`, `close`).
    Rank 0 processes them, enqueue a `LoopDecision` and awaits a future. One background `_engine_loop` per rank
    consumes the queue and executes the action. Rank 0 resolves each future when its request finishes and return
    the result back to the controller.

    Notice that vLLM `engine.step`, which is a TP collective, and the request-intake are decoupled, so a new request
    can join mid-flight, instead of waiting for the current batch to drain.

    One loop iteration, RequestDispatcher is used to dispatch requests to different ranks, and collect results.
    Take DP=1, TP=2 for example, after the controller fired generate(prompt_0) and generate(prompt_1):

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
        rank 0   request_dispatcher.process_finished_requests -> prompt_1 done? gen_future_1.set_result(Completion)
        rank 1   request_dispatcher.process_finished_requests -> no-op (tp_rank != 0, holds no futures)

    For DP>1, the requests will be routed among DPs first. See RequestDispatcher's docstring for more details.

    A weight sync rides the same loop: `pull_model_state_dict` queues a `LoopDecision(LoopAction.PULL_MODEL_STATE_DICT)` applied
    between step bursts. The engine does NOT drain in-flight requests first ("hotswap"). This behavior can be changed
    on the controller side, by blocking new requests until the engine is drained.

    Args:
        config: Generator-specific configuration.
        model_spec: TorchTitan model specification.
        model_path: Path to the HF model checkpoint.
        compile_config: Per-layer torch.compile config shared with the
            trainer so both sides compile identically.
        max_num_seqs: vLLM's upper bound on concurrently scheduled sequences (vLLM admits fewer if KV
            is tight); also sets the CUDA-graph capture sizes.
        output_dir: Structured-logger output directory.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Generator actor configuration.
        TODO: Expose a EngineConfig field to passing config to vLLM Engine"""

        parallelism: InferenceParallelismConfig = field(
            default_factory=InferenceParallelismConfig
        )
        """Parallelism configuration for the vLLM engine."""

        intra_generator_router: IntraGeneratorRouter.Config = field(
            default_factory=IntraGeneratorRouter.Config
        )
        """In-mesh DP routing config: how rank 0 partitions requests across the
        engine's data-parallel ranks (no effect when data_parallel_degree == 1,
        where there is a single DP rank)."""

        sampling: SamplingConfig = field(default_factory=SamplingConfig)
        """Default sampling parameters for generation."""

        override: OverrideConfig = field(default_factory=OverrideConfig)
        """Config overrides (e.g. ``torchtitan.overrides.fused_swiglu``) applied to
        this generator's model spec after ``update_from_config`` and before build.
        Separate from the trainer's override so the two can differ."""

        model_dtype: str = "bfloat16"
        """Data type for model weights, passed directly to vLLM (auto, float16, bfloat16, float32)."""

        gpu_memory_limit: float = 0.9
        """Fraction of GPU memory to use for the vLLM engine (0.0 to 1.0)."""

        max_num_batched_tokens: int | None = None
        """vLLM chunked-prefill chunk size: max tokens scheduled per engine step
        (prefill + decode, summed over the batch). ``None`` (default) leaves
        vLLM's own engine default in place."""

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

        reset_running_requests_on_weight_sync: bool = True
        """Affects requests ALREADY running at the pull: preempts them and recomputes their KV under
        the new weights. No effect under strict-drain (engine idle at pull time); async hot-swap only.
        Default True to avoid reusing stale-weight KV."""

        def __post_init__(self):
            # The generator runs vLLM full expert parallelism: vLLM forms the EP
            # group from all DP*TP ranks, so expert_parallel_degree must equal
            # data_parallel_degree * tensor_parallel_degree (or 1 to disable EP).
            p = self.parallelism
            full_ep = p.data_parallel_degree * p.tensor_parallel_degree
            if p.expert_parallel_degree not in (1, full_ep):
                raise ValueError(
                    f"expert_parallel_degree ({p.expert_parallel_degree}) must be 1 "
                    f"(no expert parallelism) or equal data_parallel_degree * "
                    f"tensor_parallel_degree ({full_ep}) in the generator."
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

        self._max_num_seqs = max_num_seqs

        # FULL_AND_PIECEWISE runs prefill/mixed-batch attention eager via the
        # @eager_break_during_capture decorator in rl/models/attention.py, which
        # reads VLLM_USE_BREAKABLE_CUDAGRAPH at import time -- so the env must be
        # set before register_to_vllm imports that module (#3709).
        if config.cudagraph.enable and config.cudagraph.mode == "FULL_AND_PIECEWISE":
            os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "1"

        # Register TorchTitan model + parser with vLLM
        register_to_vllm(
            model_spec,
            parallelism=config.parallelism,
            compile_config=compile_config,
            checkpoint_config=config.checkpoint,
            override=config.override,
        )

        # Set vLLM environment variables from config before any vLLM initialization
        inner_attn = model_spec.model.layers[0].attention.inner_attention
        assert isinstance(
            inner_attn,
            (VarlenAttention.Config, FlexAttention.Config),
        ), "Only varlen and flex attention backends are allowed."

        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        set_batch_invariance(config.debug.batch_invariant)
        if config.debug.batch_invariant:
            # batch_invariant_ops (via set_batch_invariance) covers
            # mm/addmm/_log_softmax/mean but not bmm; the MoE router gate lowers
            # to bmm in the vLLM inference graph, so override it generator-side.
            patch_bmm_for_batch_invariance()
            # The vLLM v2 logprob Triton kernel bypasses the aten overrides above;
            # route it through trainer's function to match the trainer exactly.
            force_logprobs_fn_for_batch_invariance()

        self._set_determinism(config.debug)

        self.model_path = model_path

        # Build vLLM engine
        enable_ep = config.parallelism.expert_parallel_degree > 1
        engine_kwargs = dict(
            # ``model`` is the path to the HF checkpoint directory. The
            # config is sourced from torchtitan's ModelSpec via
            # ``config_format=TORCHTITAN_CONFIG_FORMAT`` (no config.json
            # read), but vLLM still uses this path to locate the
            # tokenizer assets and the safetensors weight shards.
            model=model_path,
            trust_remote_code=True,
            # Use the torchtitan custom config parser (registered by
            # register_to_vllm above). It builds PretrainedConfig from
            # ModelSpec instead of reading config.json from disk.
            config_format=TORCHTITAN_CONFIG_FORMAT,
            dtype=config.model_dtype,
            tensor_parallel_size=config.parallelism.tensor_parallel_degree,
            data_parallel_size=config.parallelism.data_parallel_degree,
            # NOTE: Monarch launches the generator workers and sets the torch
            # elastic distributed env; with external_launcher, vLLM uses that
            # world to build its process groups. vLLM does not take an
            # explicit EP degree: when this boolean is set, it converts all
            # DP * TP ranks into the expert-parallel group for MoE layers.
            enable_expert_parallel=enable_ep,
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
        if config.max_num_batched_tokens is not None:
            engine_kwargs["max_num_batched_tokens"] = config.max_num_batched_tokens
        # Continuous batching requires FCFS scheduling: admission order must equal the
        # broadcast order on every rank
        engine_kwargs["scheduling_policy"] = "fcfs"
        # FA2 requires block_size to be a multiple of 256
        if not has_cuda_capability(9, 0):
            engine_kwargs["block_size"] = 256
        vllm_compilation_config = config.cudagraph.get_vllm_compilation_config(
            max_num_seqs=self._max_num_seqs,
            max_num_batched_tokens=config.max_num_batched_tokens,
        )
        if vllm_compilation_config is not None:
            engine_kwargs["compilation_config"] = vllm_compilation_config
        if config.debug.seed is not None:
            engine_kwargs["seed"] = config.debug.seed
        engine_args = EngineArgs(**engine_kwargs)

        with sl.log_trace_span("vllm_init"):
            logger.info("Initializing LLMEngine from EngineArgs...")
            # TODO(async-rl): capture engine-aggregate stats (KV-cache util, queue depth, preemptions,
            #   prefix-cache hit rate) via a `StatLoggerBase` in `from_engine_args`;
            self._engine = LLMEngine.from_engine_args(engine_args)
            logger.info("vLLM rollout engine initialized")

        self.policy_version = 0

        # --- Continuous-batching state (see the class docstring) ---
        self._rank = dist.get_rank()
        self._broadcast_group = dist.new_group(backend="gloo")  # for LoopDecisions
        self._engine_loop_condition = (
            asyncio.Condition()
        )  # Signals to wake up when there is work

        # --- Request dispatch ---
        # The dispatcher owns the DP/TP rank layout and the request dispatch /
        # completion fan-in (see its docstring).
        self._request_dispatcher = RequestDispatcher(
            rank=self._rank,
            parallelism=config.parallelism,
            broadcast_group=self._broadcast_group,
            vllm_parallel_config=self._engine.vllm_config.parallel_config,
            intra_generator_router=config.intra_generator_router,
        )

        # Engine-loop INBOX (rank 0): requests the controller submits; the loop reads them to decide.
        self._queued_generation_requests: list[GenerationRequest] = []
        self._model_state_dict_pull_request: ModelStateDictPullRequest | None = None
        self._close_request: CloseRequest | None = None

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
        routing_session_id: str,
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
            routing_session_id: Stable session key for in-mesh DP routing.
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
            # Register the future before enqueueing; the engine loop resolves it.
            generation_future = self._request_dispatcher.rank0_register_future(
                request_id, metrics_prefix
            )

            # Add the request to the queue; the engine loop will admit + process it.
            self._queued_generation_requests.append(
                GenerationRequest(
                    request_id=request_id,
                    prompt_token_ids=prompt_token_ids,
                    sampling=sampling,
                    routing_session_id=routing_session_id,
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
            # One-time dispatcher setup before the loop starts.
            self._request_dispatcher.setup()
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
                    # Rank 0 owns all futures, so it stamps the admitted (min) version for the whole decision.
                    # TODO: move under the engine_step call (register at generation_start, not admission).
                    # The way to do it is probably to change to RequestOutputKind.CUMULATIVE and mark per token.
                    if self._rank == 0:
                        self._request_dispatcher.rank0_stamp_min_policy_version(
                            decision.requests_per_dp_rank, self.policy_version
                        )
                    # Admit only this rank's DP replica slice. TP ranks in the same
                    # replica compute the same _dp_rank, so they add the identical
                    # set in the same FCFS order.
                    local_requests = decision.requests_per_dp_rank[
                        self._request_dispatcher._dp_rank
                    ]
                    if local_requests:
                        # render_cmpl is vLLM's input pipeline (tokenize is a no-op for tokenized prompts);
                        # the high-level entry stays resilient to vLLM internals vs vllm.inputs.tokens_input.
                        engine_inputs = self._engine.renderer.render_cmpl(
                            [
                                {"prompt_token_ids": request.prompt_token_ids}
                                for request in local_requests
                            ]
                        )
                        for request, engine_input in zip(
                            local_requests, engine_inputs, strict=True
                        ):
                            self._engine.add_request(
                                request_id=request.request_id,
                                prompt=engine_input,
                                params=self._build_sampling_params(request.sampling),
                            )

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
                        self._request_dispatcher.process_finished_requests(
                            request_outputs, self.policy_version
                        )
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
                # In-flight requests (on any DP rank) keep rank 0 issuing STEP.
                or self._request_dispatcher.rank0_has_pending_futures()
            )

            if self._close_request is not None:
                return LoopDecision(action=LoopAction.CLOSE, requests_per_dp_rank=[])

            # A weight pull takes priority over admitting new requests.
            if self._model_state_dict_pull_request is not None:
                return LoopDecision(
                    action=LoopAction.PULL_MODEL_STATE_DICT,
                    requests_per_dp_rank=[],
                    pull_version=self._model_state_dict_pull_request.version,
                )

            # STEP: admit whatever is queued (may be empty -> just keep stepping in-flight work).
            queued, self._queued_generation_requests = (
                self._queued_generation_requests,
                [],
            )
            return LoopDecision(
                action=LoopAction.STEP,
                requests_per_dp_rank=self._request_dispatcher.rank0_route(queued),
            )

    def _fail_outstanding_futures(self, exc: BaseException) -> None:
        """Fail every unresolved future after an exception or engine teardown."""
        self._request_dispatcher.fail_generation_futures(exc)

        if self._pull_model_state_dict_future is not None:
            if not self._pull_model_state_dict_future.done():
                self._pull_model_state_dict_future.set_exception(exc)
            self._pull_model_state_dict_future = None
            self._model_state_dict_pull_request = None

    def _build_sampling_params(self, sampling: SamplingConfig) -> SamplingParams:
        """Translate a `SamplingConfig` into vLLM `SamplingParams` (n=1).

        ``seed`` and ``stop_token_ids`` are carried on the ``SamplingConfig``
        (the controller fills ``stop_token_ids`` and the rollouter offsets
        ``seed`` per sample), so each sample in a group is a distinct ``n=1``
        request that stays diverse and bitwise-reproducible.
        """
        return SamplingParams(
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            max_tokens=sampling.max_tokens,
            n=1,  # always expects a single sample per request. Caller can call N times.
            stop_token_ids=sampling.stop_token_ids or None,
            seed=sampling.seed,
            logprobs=0,  # return only the sampled token's logprob (for the GRPO ratio)
            # Return each request's result once, when it is fully done, instead of streaming partial
            # outputs as tokens arrive.
            # TODO(async-rl): use RequestOutputKind.CUMULATIVE for exact per-token
            #   (start_token, version) boundaries; today we keep only the per-turn min/max.
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
        # Async RL uses a StorageVolume snapshot so generators do not read
        # live trainer GPU tensors while optimizer steps may be mutating them.
        # TODO(async-rl): use 2 version keys so trainer can push a new version
        # without being blocked by a generator's ongoing pull.
        model = self._get_model()
        model_sd = model.model.state_dict()
        if get_spmd_backend() == "spmd_types":
            await self._get_spmd_state_dict(model_sd, model=model)
        else:
            await ts.get_state_dict(
                "model_state_dict",
                user_state_dict=model_sd,
                strict=False,
                direct_rdma=False,
            )
        # state_dict() returns hook-produced copies for fused modules (e.g.
        # FusedQKVLinear's wqkv -> wq/wk/wv), so the in-place fill above never
        # reaches the real param. Re-apply via load_state_dict to run the merge hook.
        # Non-fused params share storage with model_sd, so reloading them is a
        # harmless self-copy; only the fused wqkv is actually rebuilt.
        # TODO: investigate can we avoid the copy and properly load fused qkv weights
        model.model.load_state_dict(model_sd, strict=False)
        self.policy_version = version
        if self.config.reset_prefix_cache_on_weight_sync:
            # TODO(async-rl): consider a `flush_kv_cache_every_n_steps` flag to force-flush every N steps
            #   (helps long generations that span many steps).
            # TODO(async-rl): salt the prefix cache per NEW rollout so a new rollout can't reuse stale-weight
            #   KV, while an in-flight rollout keeps reusing its own KV (avoids the full drop).
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

    async def _get_spmd_state_dict(self, model_sd: dict, *, model) -> None:
        """Fetch trainer-pushed weights into a spmd_types generator state dict.

        spmd_types generators hold plain local tensors, but TorchStore already
        knows how to fill DTensor state-dict entries. Wrap each local tensor as
        a DTensor using its declared SPMD layout, fetch through the normal
        state-dict path, then put the local tensors back before load_state_dict.
        """
        layouts = _fqn_to_spmd_layout(model.model)
        fused_layout_keys = sorted(
            name for name in layouts if _is_fused_qkv_state_key(name)
        )
        print(
            "[rl_weight_sync_debug] "
            f"rank={self._rank} fused_qkv_layout_keys={len(fused_layout_keys)} "
            f"sample={fused_layout_keys[:6]}",
            flush=True,
        )

        dtensor_model_sd = dict(model_sd)
        with torch.no_grad():
            for name, target in model_sd.items():
                if not isinstance(target, torch.Tensor):
                    continue

                layout = layouts.get(name)
                if layout is None:
                    # Backend-owned state such as vLLM attention scale buffers
                    # is plain replicated state and has no ShardingConfig entry.
                    # Leave it as a normal tensor for TorchStore to fill.
                    continue

                mesh = model.parallel_dims.resolve_mesh(layout.axes())
                if mesh is None:
                    continue

                placements = resolve_placements(layout, mesh)
                if _is_fused_qkv_state_key(name):
                    print(
                        "[rl_weight_sync_debug] "
                        f"rank={self._rank} wrap_fused_qkv name={name} "
                        f"local_shape={tuple(target.shape)} "
                        f"mesh_axes={mesh.mesh_dim_names} placements={placements} "
                        f"{_tensor_debug_sample(target)}",
                        flush=True,
                    )

                dtensor_model_sd[name] = DTensor.from_local(
                    target,
                    mesh,
                    placements,
                    run_check=False,
                )

        fetched_model_sd = await ts.get_state_dict(
            "model_state_dict",
            user_state_dict=dtensor_model_sd,
            strict=False,
            direct_rdma=False,
        )
        fetched_fused_qkv_keys = sorted(
            name for name in fetched_model_sd if _is_fused_qkv_state_key(name)
        )
        print(
            "[rl_weight_sync_debug] "
            f"rank={self._rank} fetched_fused_qkv_keys="
            f"{len(fetched_fused_qkv_keys)} sample={fetched_fused_qkv_keys[:6]}",
            flush=True,
        )

        with torch.no_grad():
            for name, value in dtensor_model_sd.items():
                if isinstance(value, DTensor):
                    local_value = value.to_local()
                    if _is_fused_qkv_state_key(name):
                        print(
                            "[rl_weight_sync_debug] "
                            f"rank={self._rank} loaded_fused_qkv name={name} "
                            f"local_shape={tuple(local_value.shape)} "
                            f"mesh_axes={value.device_mesh.mesh_dim_names} "
                            f"placements={value.placements} "
                            f"{_tensor_debug_sample(local_value)}",
                            flush=True,
                        )
                    model_sd[name] = local_value

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

        # Stop the result-drain task on rank 0.
        await self._request_dispatcher.shutdown()

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
    routing_session_id: str


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
    """A generation request's future the loop resolves with its `Completion`."""

    future: asyncio.Future[Completion]
    metrics_prefix: str
    """Namespaces this generation's metrics (e.g. `generator` vs `validation_generator`)."""
    min_policy_version: int = field(init=False)
    """Policy version the request was admitted (sampled) under; the max is read at finish (see `Completion`)."""


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

    requests_per_dp_rank: list[list[GenerationRequest]] | None = None
    # Per-DP-rank requests to admit before a STEP burst; index == DP rank, fixed
    # length data_parallel_degree. Each rank admits only its own DP-rank slice.
    # (empty unless any queued)

    pull_version: int | None = None
    # set iff action is PULL_MODEL_STATE_DICT
