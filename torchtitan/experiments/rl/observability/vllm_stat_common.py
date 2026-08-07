# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared building blocks for the RL generator's vLLM ``StatLoggerBase``s.
"""

from __future__ import annotations

import logging
import os
import socket
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm.v1.metrics.loggers import StatLoggerBase

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.metrics.stats import (
        IterationStats,
        MultiModalCacheStats,
        SchedulerStats,
    )

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class StatIdentity:
    """Per-engine identity: whether to log, and the tags every row carries."""

    should_log: bool
    # Numeric values (rank, dp_rank, ...) are ``int``; the rest (model_name,
    # hostname, ...) are ``str``.
    attributes: dict[str, str | int]


@dataclass
class StatLoggerContext:
    """Per-engine context the generator injects into its stat loggers."""

    rank: int
    tp_rank: int
    dp_rank: int
    generator_name: str
    output_dir: str


@dataclass
class FinishedRequestStats:
    """Per-finished-request latencies for one engine step (milliseconds)."""

    decode_time_ms: float
    queue_time_ms: float
    e2e_latency_ms: float


@dataclass
class StepStats:
    """Backend-neutral snapshot of one vLLM engine step."""

    kv_cache_usage: float | None
    num_running_reqs: int | None
    num_waiting_reqs: int | None
    prefix_queries: int
    prefix_hits: int
    num_generation_tokens: int
    num_prompt_tokens: int
    num_cached_prompt_tokens: int
    num_preempted_reqs: int
    ttft_ms: list[float] = field(default_factory=list)
    itl_ms: list[float] = field(default_factory=list)
    finished: list[FinishedRequestStats] = field(default_factory=list)


def build_stat_identity(
    vllm_config: VllmConfig,
    engine_index: int,
    context: StatLoggerContext,
    extra_attributes: dict[str, str | int] | None = None,
) -> StatIdentity:
    """Compute the per-engine identity."""
    rank = context.rank
    tp_rank = context.tp_rank
    dp_rank = context.dp_rank

    # LOCAL_RANK / WORLD_SIZE are standard torch.distributed env vars (tags only).
    attributes: dict[str, str | int] = {
        "model_name": vllm_config.model_config.model,
        "hostname": socket.gethostname(),
        "rank": rank,
        "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
        "world_size": int(os.environ.get("WORLD_SIZE", 1)),
        "dp_rank": dp_rank,
        "tp_rank": tp_rank,
        "engine_index": engine_index,
    }
    attributes["generator_name"] = context.generator_name
    if extra_attributes:
        attributes.update(extra_attributes)

    # All TP ranks in a DP replica see identical engine-aggregate stats,
    # so only tp_rank==0 needs to log.
    return StatIdentity(should_log=tp_rank == 0, attributes=attributes)


def extract_step_stats(
    scheduler_stats: SchedulerStats | None,
    iteration_stats: IterationStats | None,
) -> StepStats:
    """Read one engine step's vLLM stats into a backend-neutral ``StepStats``.
    """
    kv_cache_usage: float | None = None
    num_running: int | None = None
    num_waiting: int | None = None
    prefix_queries = 0
    prefix_hits = 0
    if scheduler_stats is not None:
        kv_cache_usage = scheduler_stats.kv_cache_usage
        num_running = scheduler_stats.num_running_reqs
        num_waiting = scheduler_stats.num_waiting_reqs
        prefix = scheduler_stats.prefix_cache_stats
        if prefix is not None:
            prefix_queries = prefix.queries
            prefix_hits = prefix.hits

    gen_tokens = 0
    prompt_tokens = 0
    cached_prompt_tokens = 0
    num_preempted = 0
    ttft_ms: list[float] = []
    itl_ms: list[float] = []
    finished: list[FinishedRequestStats] = []
    if iteration_stats is not None:
        gen_tokens = iteration_stats.num_generation_tokens
        prompt_tokens = iteration_stats.prompt_token_stats.total
        cached_prompt_tokens = iteration_stats.prompt_token_stats.cached_tokens
        num_preempted = iteration_stats.num_preempted_reqs
        ttft_ms = [t * 1000 for t in iteration_stats.time_to_first_tokens_iter]
        itl_ms = [t * 1000 for t in iteration_stats.inter_token_latencies_iter]
        finished = [
            FinishedRequestStats(
                decode_time_ms=f.decode_time * 1000,
                queue_time_ms=f.queued_time * 1000,
                e2e_latency_ms=f.e2e_latency * 1000,
            )
            for f in iteration_stats.finished_requests
        ]

    return StepStats(
        kv_cache_usage=kv_cache_usage,
        num_running_reqs=num_running,
        num_waiting_reqs=num_waiting,
        prefix_queries=prefix_queries,
        prefix_hits=prefix_hits,
        num_generation_tokens=gen_tokens,
        num_prompt_tokens=prompt_tokens,
        num_cached_prompt_tokens=cached_prompt_tokens,
        num_preempted_reqs=num_preempted,
        ttft_ms=ttft_ms,
        itl_ms=itl_ms,
        finished=finished,
    )


class VllmStatLoggerBase(StatLoggerBase):
    """Base for the generator's vLLM stat loggers."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_index: int = 0,
        *,
        context: StatLoggerContext,
        extra_attributes: dict[str, str | int] | None = None,
    ) -> None:
        """Overrides ``StatLoggerBase.__init__``.
        """
        self._engine_index = engine_index
        identity = build_stat_identity(
            vllm_config, engine_index, context, extra_attributes=extra_attributes
        )
        self._output_dir: str = context.output_dir
        self._attributes: dict[str, str | int] = identity.attributes
        # a static gate: whether this logger should log at all. Only True for
        # tp_rank == 0's logger.
        self._should_log: bool = identity.should_log
        # a dynamic gate: flips True when this engine should log, and the underlying
        # backend is up. Flips False when the backend is down so the logger stays
        # silent.
        self._enabled: bool = False

    def record(
        self,
        scheduler_stats: SchedulerStats | None,
        iteration_stats: IterationStats | None,
        mm_cache_stats: MultiModalCacheStats | None = None,
        engine_idx: int = 0,
    ) -> None:
        """Overrides ``StatLoggerBase.record``. Extract one engine step's stats
        and hand them to ``_accumulate``.

        vLLM calls this every engine step, so it runs on the hot path and must
        never raise into the engine loop.
        """
        if not self._enabled:
            return
        try:
            self._accumulate(extract_step_stats(scheduler_stats, iteration_stats))
        except Exception as e:
            logger.warning(
                "%s disabled for the rest of the run: record() failed "
                "(vLLM stats schema drift?): %s",
                type(self).__name__,
                e,
            )
            self._enabled = False

    def log_engine_initialized(self) -> None:
        """Overrides ``StatLoggerBase.log_engine_initialized``.

        vLLM calls this once per engine after construction.
        """
        if self._enabled:
            logger.info(
                "%s active: engine_index=%d (flush cadence: VLLM_LOG_STATS_INTERVAL)",
                type(self).__name__,
                self._engine_index,
            )

    def _accumulate(self, step: StepStats) -> None:
        """Fold one step's ``StepStats`` into the subclass's window / instruments.

        Called from ``record()`` on every engine step; it only buffers in memory and
        never writes to the backend. The accumulated window is flushed by ``log()``,
        which vLLM calls on the ``VLLM_LOG_STATS_INTERVAL`` cadence (time-based,
        default 10s) -- so the backend write rate is one flush per interval,
        independent of the per-step ``_accumulate`` rate.
        """
        raise NotImplementedError
