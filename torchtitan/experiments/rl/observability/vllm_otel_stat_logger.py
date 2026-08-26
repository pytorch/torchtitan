# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenTelemetry (OTLP) ``StatLoggerBase`` for the RL generator's vLLM engine.

The logger stays inert unless:
  - ``OTEL_METRICS_EXPORTER`` is set to ``jsonl``; or
  - ``OTEL_METRICS_EXPORTER`` is set to ``otlp`` and an OTLP endpoint is set.
"""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING

from vllm import envs

from torchtitan.experiments.rl.observability.vllm_stat_common import (
    StatLoggerContext,
    StepStats,
    VllmStatLoggerBase,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from opentelemetry.metrics import CallbackOptions, Meter, Observation
    from vllm.config import VllmConfig

logger: logging.Logger = logging.getLogger(__name__)


def _compact_json(metrics_data) -> str:
    # One JSON object per flush on a single line (indent=None) so each row is
    # greppable in a log/file; the SDK default is pretty-printed multi-line.
    return metrics_data.to_json(indent=None) + "\n"


class VllmOtelStatLogger(VllmStatLoggerBase):
    """Per-engine vLLM stat logger that exports OTLP metrics to a collector."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_index: int = 0,
        *,
        context: StatLoggerContext,
    ) -> None:
        super().__init__(vllm_config, engine_index, context=context)

        self._kv_cache_usage_last = 0.0
        self._num_running_last = 0
        self._num_waiting_last = 0

        self._provider = None
        if self._should_log:
            self._init_otel(self._attributes)

    def _init_otel(self, resource_attrs: dict[str, str | int]) -> None:
        """Initialize the OTLP export pipeline."""
        configured_exporter = os.environ.get("OTEL_METRICS_EXPORTER", "none")
        exporter_kind = configured_exporter.strip().lower()
        if exporter_kind == "none":
            logger.info(
                "VllmOtelStatLogger inactive: set OTEL_METRICS_EXPORTER to "
                "jsonl or otlp"
            )
            return
        if exporter_kind not in ("jsonl", "otlp"):
            logger.warning(
                "VllmOtelStatLogger inactive: unsupported "
                "OTEL_METRICS_EXPORTER=%r; expected one of: jsonl, none, otlp",
                configured_exporter,
            )
            return
        if os.environ.get("OTEL_SDK_DISABLED", "").strip().lower() == "true":
            logger.warning(
                "VllmOtelStatLogger inactive: OTEL_SDK_DISABLED is set to true"
            )
            return
        if exporter_kind == "jsonl" and not self._output_dir:
            logger.warning(
                "VllmOtelStatLogger inactive: OTEL_METRICS_EXPORTER=jsonl "
                "requires an output directory"
            )
            return
        if exporter_kind == "otlp":
            endpoint = os.environ.get(
                "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"
            ) or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
            if not endpoint:
                logger.warning(
                    "VllmOtelStatLogger inactive: OTEL_METRICS_EXPORTER=otlp "
                    "requires OTEL_EXPORTER_OTLP_ENDPOINT or "
                    "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"
                )
                return
            protocol = (
                os.environ.get(
                    "OTEL_EXPORTER_OTLP_METRICS_PROTOCOL",
                    os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf"),
                )
                .strip()
                .lower()
            )
            if protocol != "http/protobuf":
                logger.warning(
                    "VllmOtelStatLogger inactive: configured OTLP protocol %r is "
                    "unsupported; this logger uses the OTLP HTTP/protobuf exporter "
                    "and requires http/protobuf",
                    protocol,
                )
                return

        try:
            from opentelemetry.metrics import Observation
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
            from opentelemetry.sdk.resources import Resource

            if exporter_kind == "jsonl":
                from opentelemetry.sdk.metrics.export import ConsoleMetricExporter

                # Save metrics to ``<output_dir>/vllm_metrics/``. Useful for
                # local development where a collector is not available.
                rank = self._attributes.get("rank", 0)
                label = str(self._attributes.get("generator_name", "gen"))
                safe_label = re.sub(r"[^A-Za-z0-9._-]", "_", label)
                path = os.path.join(
                    self._output_dir,
                    "vllm_metrics",
                    f"{safe_label}.rank{rank}.jsonl",
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
            self._provider = MeterProvider(
                metric_readers=[reader],
                resource=Resource.create(resource_attrs),
            )
            meter = self._provider.get_meter("torchtitan.experiments.rl.vllm")
            self._create_counters(meter)
            self._create_histograms(meter)
            self._create_gauges(meter, Observation)
            self._enabled = True
        except Exception as e:
            logger.warning("VllmOtelStatLogger disabled: metrics setup failed: %s", e)
            self._enabled = False

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

    def _accumulate(self, step: StepStats) -> None:
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

    def log(self) -> None:
        """Leave vLLM's synchronous logging hook empty.

        The OpenTelemetry metric reader exports on its background thread.
        """
        pass

    def log_engine_initialized(self) -> None:
        """Log the active export cadence after vLLM initializes the engine."""
        if self._enabled:
            logger.info(
                "%s active: engine_index=%d "
                "(export cadence: VLLM_LOG_STATS_INTERVAL=%gs)",
                type(self).__name__,
                self._engine_index,
                envs.VLLM_LOG_STATS_INTERVAL,
            )
