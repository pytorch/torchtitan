# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenTelemetry (OTLP) ``StatLoggerBase`` for the RL generator's vLLM engine.

See ``torchtitan/experiments/rl/observability/metrics/README.md`` for setup and
the exported metric definitions.

The logger stays inert unless:
  - ``OTEL_METRICS_EXPORTER`` is set to ``jsonl``; or
  - ``OTEL_METRICS_EXPORTER`` is set to ``otlp`` and an OTLP endpoint is set.
"""

from __future__ import annotations

import logging
import os
import re
import socket
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from opentelemetry.util.types import AttributeValue
from vllm import envs
from vllm.v1.metrics.loggers import StatLoggerBase

from torchtitan.config import Configurable

if TYPE_CHECKING:
    from collections.abc import Iterable

    from opentelemetry.metrics import CallbackOptions, Meter, Observation
    from vllm.config import VllmConfig
    from vllm.v1.metrics.stats import (
        IterationStats,
        MultiModalCacheStats,
        SchedulerStats,
    )

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class StatLoggerContext:
    """Per-engine context the generator injects into its stat loggers."""

    rank: int
    tp_rank: int
    dp_rank: int
    generator_name: str
    output_dir: str


@dataclass
class _FinishedRequestStats:
    """Per-finished-request latencies for one engine step (milliseconds)."""

    decode_time_ms: float
    queue_time_ms: float
    e2e_latency_ms: float


@dataclass
class _StepStats:
    """Snapshot of one vLLM engine step."""

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
    finished: list[_FinishedRequestStats] = field(default_factory=list)


def _extract_step_stats(
    scheduler_stats: SchedulerStats | None,
    iteration_stats: IterationStats | None,
) -> _StepStats:
    """Read one engine step's vLLM stats."""
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
    finished: list[_FinishedRequestStats] = []
    if iteration_stats is not None:
        gen_tokens = iteration_stats.num_generation_tokens
        prompt_tokens = iteration_stats.prompt_token_stats.total
        cached_prompt_tokens = iteration_stats.prompt_token_stats.cached_tokens
        num_preempted = iteration_stats.num_preempted_reqs
        ttft_ms = [t * 1000 for t in iteration_stats.time_to_first_tokens_iter]
        itl_ms = [t * 1000 for t in iteration_stats.inter_token_latencies_iter]
        finished = [
            _FinishedRequestStats(
                decode_time_ms=request.decode_time * 1000,
                queue_time_ms=request.queued_time * 1000,
                e2e_latency_ms=request.e2e_latency * 1000,
            )
            for request in iteration_stats.finished_requests
        ]

    return _StepStats(
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


def _compact_json(metrics_data) -> str:
    # One JSON object per flush on a single line (indent=None) so each row is
    # greppable in a log/file; the SDK default is pretty-printed multi-line.
    return metrics_data.to_json(indent=None) + "\n"


class VllmOtelStatLogger(Configurable, StatLoggerBase):
    """Per-engine vLLM stat logger that exports OpenTelemetry metrics.

    Metric selection and stat extraction are based on vLLM's ``PrometheusStatLogger``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        extra_resource_attributes: dict[str, AttributeValue] = field(
            default_factory=dict
        )
        """Additional OpenTelemetry resource attributes. Values are applied after
        automatically generated attributes and override them when keys collide."""

    def __init__(
        self,
        config: Config,
        vllm_config: VllmConfig,
        engine_index: int = 0,
        *,
        context: StatLoggerContext,
    ) -> None:
        self._config = config
        self._enabled = False

        self._kv_cache_usage_last = 0.0
        self._num_running_last = 0
        self._num_waiting_last = 0

        configured_exporter = os.environ.get("OTEL_METRICS_EXPORTER", "none")
        exporter_kind = configured_exporter.strip().lower()
        if exporter_kind == "none":
            logger.info(
                "VllmOtelStatLogger inactive: set OTEL_METRICS_EXPORTER to "
                "jsonl or otlp if you want to record vLLM metrics"
            )
            return
        if exporter_kind not in ("jsonl", "otlp"):
            raise ValueError(
                "unsupported OTEL_METRICS_EXPORTER="
                f"{configured_exporter!r}; expected one of: jsonl, none, otlp"
            )
        if os.environ.get("OTEL_SDK_DISABLED", "").strip().lower() == "true":
            logger.warning(
                "VllmOtelStatLogger inactive: OTEL_SDK_DISABLED is set to true"
            )
            return
        if exporter_kind == "jsonl" and not context.output_dir:
            raise ValueError(
                "OTEL_METRICS_EXPORTER=jsonl requires an output directory. "
                f"Found {context.output_dir=}"
            )
        if exporter_kind == "otlp":
            endpoint = os.environ.get(
                "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"
            ) or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
            if not endpoint:
                raise ValueError(
                    "OTEL_METRICS_EXPORTER=otlp "
                    "requires OTEL_EXPORTER_OTLP_ENDPOINT or "
                    "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"
                )
            protocol = (
                os.environ.get(
                    "OTEL_EXPORTER_OTLP_METRICS_PROTOCOL",
                    os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf"),
                )
                .strip()
                .lower()
            )
            if protocol != "http/protobuf":
                raise ValueError(
                    f"configured OTLP protocol {protocol!r} is unsupported; "
                    "this logger uses the OTLP HTTP/protobuf exporter and requires "
                    "http/protobuf"
                )

        try:
            from opentelemetry.metrics import Observation
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
            from opentelemetry.sdk.resources import Resource

            if exporter_kind == "jsonl":
                from opentelemetry.sdk.metrics.export import ConsoleMetricExporter

                # Save metrics to ``<output_dir>/vllm_metrics/``. Useful for
                # local development where a collector is not available.
                safe_label = re.sub(r"[^A-Za-z0-9._-]", "_", context.generator_name)
                path = os.path.join(
                    context.output_dir,
                    "vllm_metrics",
                    f"{safe_label}.rank{context.rank}.jsonl",
                )
                os.makedirs(os.path.dirname(path), exist_ok=True)
                out = open(path, "a", buffering=1)
                exporter = ConsoleMetricExporter(out=out, formatter=_compact_json)
            else:
                from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                    OTLPMetricExporter,
                )

                exporter = OTLPMetricExporter()

            # Export on the OpenTelemetry SDK's background thread. In particular,
            # never perform network I/O from vLLM's synchronous log() callback.
            reader = PeriodicExportingMetricReader(
                exporter,
                export_interval_millis=envs.VLLM_LOG_STATS_INTERVAL * 1000,
            )
            provider = MeterProvider(
                metric_readers=[reader],
                resource=Resource.create(
                    self._build_resource_attributes(vllm_config, context)
                ),
            )
            meter = provider.get_meter("torchtitan.experiments.rl.vllm")
            self._create_counters(meter)
            self._create_histograms(meter)
            self._create_gauges(meter, Observation)
            self._enabled = True
        except Exception as e:
            logger.warning("VllmOtelStatLogger disabled: metrics setup failed: %s", e)
            self._enabled = False

    def _build_resource_attributes(
        self, vllm_config: VllmConfig, context: StatLoggerContext
    ) -> dict[str, AttributeValue]:
        """Build resource attributes."""
        attributes: dict[str, AttributeValue] = {
            "model_name": vllm_config.model_config.model,
            "hostname": socket.gethostname(),
            "rank": context.rank,
            "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
            "world_size": int(os.environ.get("WORLD_SIZE", 1)),
            "dp_rank": context.dp_rank,
            "tp_rank": context.tp_rank,
            "generator_name": context.generator_name,
        }
        attributes.update(self._config.extra_resource_attributes)
        return attributes

    def _create_counters(self, meter: Meter) -> None:
        """Monotonic counters; the backend derives throughput/rates via rate()."""
        self._c_gen_tokens = meter.create_counter(
            "vllm.generation_tokens", unit="token"
        )
        self._c_prompt_tokens = meter.create_counter("vllm.prompt_tokens", unit="token")
        self._c_cached_prompt_tokens = meter.create_counter(
            "vllm.cached_prompt_tokens", unit="token"
        )
        self._c_preempted = meter.create_counter(
            "vllm.preempted_requests", unit="request"
        )
        self._c_finished = meter.create_counter(
            "vllm.finished_requests", unit="request"
        )
        # Prefix-cache hit rate = rate(hits) / rate(queries) at query time.
        self._c_prefix_queries = meter.create_counter("vllm.prefix_cache_queries")
        self._c_prefix_hits = meter.create_counter("vllm.prefix_cache_hits")

    def _create_histograms(self, meter: Meter) -> None:
        """Latency histograms; the backend computes percentiles (p50/p95/p99)."""
        self._h_ttft = meter.create_histogram("vllm.time_to_first_token", unit="ms")
        self._h_itl = meter.create_histogram("vllm.inter_token_latency", unit="ms")
        self._h_decode = meter.create_histogram("vllm.decode_time", unit="ms")
        self._h_queue = meter.create_histogram("vllm.queue_time", unit="ms")
        self._h_e2e = meter.create_histogram("vllm.e2e_latency", unit="ms")

    def _create_gauges(self, meter: Meter, observation_cls: type[Observation]) -> None:
        """Observable gauges report the last-seen scheduler state at collection."""

        def kv_cb(_options: CallbackOptions) -> Iterable[Observation]:
            return (observation_cls(self._kv_cache_usage_last),)

        def running_cb(_options: CallbackOptions) -> Iterable[Observation]:
            return (observation_cls(self._num_running_last),)

        def waiting_cb(_options: CallbackOptions) -> Iterable[Observation]:
            return (observation_cls(self._num_waiting_last),)

        meter.create_observable_gauge("vllm.kv_cache_usage", callbacks=[kv_cb])
        meter.create_observable_gauge(
            "vllm.num_running_requests", callbacks=[running_cb], unit="request"
        )
        meter.create_observable_gauge(
            "vllm.num_waiting_requests", callbacks=[waiting_cb], unit="request"
        )

    def record(
        self,
        scheduler_stats: SchedulerStats | None,
        iteration_stats: IterationStats | None,
        mm_cache_stats: MultiModalCacheStats | None = None,
        engine_idx: int = 0,
    ) -> None:
        """Record one vLLM engine step without propagating telemetry errors."""
        if not self._enabled:
            return
        try:
            step = _extract_step_stats(scheduler_stats, iteration_stats)
            if step.kv_cache_usage is not None:
                self._kv_cache_usage_last = step.kv_cache_usage
            if step.num_running_reqs is not None:
                self._num_running_last = step.num_running_reqs
            if step.num_waiting_reqs is not None:
                self._num_waiting_last = step.num_waiting_reqs

            if step.prefix_queries:
                self._c_prefix_queries.add(step.prefix_queries)
            if step.prefix_hits:
                self._c_prefix_hits.add(step.prefix_hits)

            self._c_gen_tokens.add(step.num_generation_tokens)
            self._c_prompt_tokens.add(step.num_prompt_tokens)
            self._c_cached_prompt_tokens.add(step.num_cached_prompt_tokens)
            self._c_preempted.add(step.num_preempted_reqs)
            for ttft in step.ttft_ms:
                self._h_ttft.record(ttft)
            for itl in step.itl_ms:
                self._h_itl.record(itl)
            for finished in step.finished:
                self._c_finished.add(1)
                self._h_decode.record(finished.decode_time_ms)
                self._h_queue.record(finished.queue_time_ms)
                self._h_e2e.record(finished.e2e_latency_ms)
        except Exception as e:
            logger.warning(
                "%s disabled for the rest of the run: record() failed "
                "(vLLM stats schema drift?): %s",
                type(self).__name__,
                e,
            )
            self._enabled = False

    def log(self) -> None:
        """Leave vLLM's synchronous logging hook empty.

        The OpenTelemetry metric reader exports on a background thread so exporter
        I/O does not block vLLM's generation path.
        """
        pass

    def log_engine_initialized(self) -> None:
        """Log the active export cadence after vLLM initializes the engine."""
        if self._enabled:
            logger.info(
                "%s active (export cadence: VLLM_LOG_STATS_INTERVAL=%gs)",
                type(self).__name__,
                envs.VLLM_LOG_STATS_INTERVAL,
            )
