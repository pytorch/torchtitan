# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import json
import logging
import os
from types import SimpleNamespace

import pytest

from torchtitan.experiments.rl.observability.vllm_otel_stat_logger import (
    _extract_step_stats,
    StatLoggerContext,
    VllmOtelStatLogger,
)


def _vllm_config(*, tp_size=1, dp_rank=0, model="test-model"):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp_size, data_parallel_rank=dp_rank
        ),
        model_config=SimpleNamespace(model=model),
    )


def _context(rank=0, tp_rank=0, dp_rank=0, generator_name="gen", output_dir="/tmp"):
    return StatLoggerContext(
        rank=rank,
        tp_rank=tp_rank,
        dp_rank=dp_rank,
        generator_name=generator_name,
        output_dir=output_dir,
    )


@pytest.fixture
def no_otel_env(monkeypatch):
    for key in tuple(os.environ):
        if key.startswith("OTEL_"):
            monkeypatch.delenv(key)
    monkeypatch.delenv("VLLM_LOG_STATS_INTERVAL", raising=False)


@pytest.fixture
def meter_providers(monkeypatch):
    from opentelemetry.sdk import metrics as sdk_metrics

    meter_provider_cls = sdk_metrics.MeterProvider
    providers = []

    def create_meter_provider(*args, **kwargs):
        provider = meter_provider_cls(*args, **kwargs)
        providers.append(provider)
        return provider

    monkeypatch.setattr(sdk_metrics, "MeterProvider", create_meter_provider)
    yield providers
    for provider in providers:
        provider.shutdown(timeout_millis=1000)


def test_context_populates_resource_attributes(
    no_otel_env, monkeypatch, tmp_path, meter_providers
):
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", "jsonl")
    log = VllmOtelStatLogger.Config().build(
        vllm_config=_vllm_config(model="m"),
        engine_index=3,
        context=_context(
            rank=5,
            tp_rank=0,
            dp_rank=2,
            generator_name="gen-a",
            output_dir=str(tmp_path),
        ),
    )

    attributes = meter_providers[0]._sdk_config.resource.attributes
    assert attributes["rank"] == 5
    assert attributes["tp_rank"] == 0
    assert attributes["dp_rank"] == 2
    assert attributes["model_name"] == "m"
    assert attributes["generator_name"] == "gen-a"


def test_context_reads_distributed_env_tags(
    no_otel_env, monkeypatch, tmp_path, meter_providers
):
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", "jsonl")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "8")

    log = VllmOtelStatLogger.Config().build(
        vllm_config=_vllm_config(),
        context=_context(rank=3, tp_rank=1, dp_rank=1, output_dir=str(tmp_path)),
    )

    attributes = meter_providers[0]._sdk_config.resource.attributes
    assert attributes["local_rank"] == 1
    assert attributes["world_size"] == 8


def test_extra_resource_attributes_support_arrays_and_overrides(
    no_otel_env, monkeypatch, tmp_path, meter_providers
):
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", "jsonl")
    VllmOtelStatLogger.Config(
        extra_resource_attributes={
            "custom.columns": ["request_id", "latency_ms"],
            "model_name": "overridden-model",
        }
    ).build(
        vllm_config=_vllm_config(model="original-model"),
        context=_context(output_dir=str(tmp_path)),
    )

    attributes = meter_providers[0]._sdk_config.resource.attributes
    assert attributes["custom.columns"] == ("request_id", "latency_ms")
    assert attributes["model_name"] == "overridden-model"


def test_extract_step_stats_both_none():
    step = _extract_step_stats(None, None)

    assert step.kv_cache_usage is None
    assert step.num_running_reqs is None
    assert step.num_waiting_reqs is None
    assert step.prefix_queries == 0
    assert step.prefix_hits == 0
    assert step.num_generation_tokens == 0
    assert step.ttft_ms == []
    assert step.itl_ms == []
    assert step.finished == []


def test_extract_step_stats_populated_converts_seconds_to_ms():
    scheduler = SimpleNamespace(
        kv_cache_usage=0.5,
        num_running_reqs=4,
        num_waiting_reqs=2,
        prefix_cache_stats=SimpleNamespace(queries=10, hits=7),
    )
    iteration = SimpleNamespace(
        num_generation_tokens=12,
        prompt_token_stats=SimpleNamespace(total=100, cached_tokens=30),
        num_preempted_reqs=1,
        time_to_first_tokens_iter=[0.012],
        inter_token_latencies_iter=[0.01, 0.02],
        finished_requests=[
            SimpleNamespace(decode_time=0.03, queued_time=0.005, e2e_latency=0.05)
        ],
    )

    step = _extract_step_stats(scheduler, iteration)

    assert step.kv_cache_usage == 0.5
    assert step.num_running_reqs == 4
    assert step.num_waiting_reqs == 2
    assert step.prefix_queries == 10
    assert step.prefix_hits == 7
    assert step.num_generation_tokens == 12
    assert step.num_prompt_tokens == 100
    assert step.num_cached_prompt_tokens == 30
    assert step.num_preempted_reqs == 1
    assert step.ttft_ms == pytest.approx([12.0])
    assert step.itl_ms == pytest.approx([10.0, 20.0])
    assert len(step.finished) == 1
    assert step.finished[0].decode_time_ms == pytest.approx(30.0)
    assert step.finished[0].queue_time_ms == pytest.approx(5.0)
    assert step.finished[0].e2e_latency_ms == pytest.approx(50.0)


def test_extract_step_stats_prefix_cache_none():
    scheduler = SimpleNamespace(
        kv_cache_usage=0.1,
        num_running_reqs=1,
        num_waiting_reqs=0,
        prefix_cache_stats=None,
    )

    step = _extract_step_stats(scheduler, None)

    assert step.prefix_queries == 0
    assert step.prefix_hits == 0
    assert step.kv_cache_usage == 0.1


def test_inert_without_exporter(no_otel_env):
    log = VllmOtelStatLogger.Config().build(
        vllm_config=_vllm_config(),
        engine_index=0,
        context=_context(rank=0, tp_rank=0, dp_rank=0),
    )
    assert log._enabled is False
    # No-ops that must never raise and never touch (uncreated) instruments.
    log.record(None, None)
    log.log()


def test_logger_does_not_gate_on_tp_rank(
    no_otel_env, monkeypatch, tmp_path, meter_providers
):
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", "jsonl")
    log = VllmOtelStatLogger.Config().build(
        vllm_config=_vllm_config(tp_size=2),
        engine_index=0,
        context=_context(rank=1, tp_rank=1, dp_rank=0, output_dir=str(tmp_path)),
    )
    assert log._enabled is True
    assert len(meter_providers) == 1


def test_exporter_none_is_inert(monkeypatch):
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", "none")
    log = VllmOtelStatLogger.Config().build(
        vllm_config=_vllm_config(),
        engine_index=0,
        context=_context(rank=0, tp_rank=0, dp_rank=0),
    )
    assert log._enabled is False


def test_endpoint_without_exporter_is_inert(no_otel_env, monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")

    log = VllmOtelStatLogger.Config().build(
        vllm_config=_vllm_config(), context=_context()
    )

    assert log._enabled is False


def test_sdk_disabled_is_inert(no_otel_env, monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", "jsonl")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")

    with caplog.at_level(logging.WARNING):
        log = VllmOtelStatLogger.Config().build(
            vllm_config=_vllm_config(),
            context=_context(output_dir=str(tmp_path)),
        )

    assert log._enabled is False
    assert "OTEL_SDK_DISABLED is set to true" in caplog.text
    assert not (tmp_path / "vllm_metrics").exists()


@pytest.mark.parametrize("exporter", ["console", "jsonl,otlp", "unknown"])
def test_unsupported_exporter_raises(no_otel_env, monkeypatch, exporter):
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", exporter)

    with pytest.raises(ValueError, match="unsupported OTEL_METRICS_EXPORTER"):
        VllmOtelStatLogger.Config().build(
            vllm_config=_vllm_config(), context=_context()
        )


def test_jsonl_requires_output_dir(no_otel_env, monkeypatch):
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", "jsonl")

    with pytest.raises(ValueError, match="requires an output directory.*output_dir=''"):
        VllmOtelStatLogger.Config().build(
            vllm_config=_vllm_config(),
            context=_context(output_dir=""),
        )


def test_otlp_requires_endpoint(no_otel_env, monkeypatch):
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", "otlp")

    with pytest.raises(ValueError, match="requires OTEL_EXPORTER_OTLP_ENDPOINT"):
        VllmOtelStatLogger.Config().build(
            vllm_config=_vllm_config(), context=_context()
        )


def test_otlp_requires_http_protobuf(no_otel_env, monkeypatch):
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", "otlp")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")

    with pytest.raises(ValueError, match="uses the OTLP HTTP/protobuf exporter"):
        VllmOtelStatLogger.Config().build(
            vllm_config=_vllm_config(), context=_context()
        )


def test_jsonl_export_is_asynchronous(
    no_otel_env, monkeypatch, tmp_path, caplog, meter_providers
):
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", "jsonl")
    monkeypatch.setenv("OTEL_METRIC_EXPORT_TIMEOUT", "1234")
    monkeypatch.setenv("VLLM_LOG_STATS_INTERVAL", "600")
    log = VllmOtelStatLogger.Config().build(
        vllm_config=_vllm_config(),
        context=_context(output_dir=str(tmp_path)),
    )

    assert log._enabled is True
    provider = meter_providers[0]
    reader = provider._metric_readers[0]
    assert reader._daemon_thread is not None
    assert reader._export_interval_millis == 600000
    assert reader._export_timeout_millis == 1234

    with caplog.at_level(logging.INFO):
        log.log_engine_initialized()
    assert "VLLM_LOG_STATS_INTERVAL=600" in caplog.text

    # vLLM calls log() from LLMEngine.step(). It must remain a no-op instead of
    # performing synchronous exporter I/O on the generation path.
    monkeypatch.setattr(
        provider,
        "force_flush",
        lambda *args, **kwargs: pytest.fail("force_flush called from log()"),
    )
    log.log()


def test_jsonl_export_writes_recorded_metrics(
    no_otel_env, monkeypatch, tmp_path, meter_providers
):
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", "jsonl")
    monkeypatch.setenv("VLLM_LOG_STATS_INTERVAL", "600")
    log = VllmOtelStatLogger.Config().build(
        vllm_config=_vllm_config(model="test-model"),
        context=_context(
            rank=5,
            tp_rank=0,
            dp_rank=2,
            generator_name="gen-a",
            output_dir=str(tmp_path),
        ),
    )
    scheduler = SimpleNamespace(
        kv_cache_usage=0.5,
        num_running_reqs=4,
        num_waiting_reqs=2,
        prefix_cache_stats=SimpleNamespace(queries=10, hits=7),
    )
    iteration = SimpleNamespace(
        num_generation_tokens=12,
        prompt_token_stats=SimpleNamespace(total=100, cached_tokens=30),
        num_preempted_reqs=1,
        time_to_first_tokens_iter=[0.012],
        inter_token_latencies_iter=[0.01, 0.02],
        finished_requests=[
            SimpleNamespace(decode_time=0.03, queued_time=0.005, e2e_latency=0.05)
        ],
    )

    log.record(scheduler, iteration)
    meter_providers[0].force_flush(timeout_millis=1000)

    output_path = tmp_path / "vllm_metrics" / "gen-a.rank5.jsonl"
    rows = output_path.read_text().splitlines()
    assert len(rows) == 1
    resource_metrics = json.loads(rows[0])["resource_metrics"][0]
    resource_attributes = resource_metrics["resource"]["attributes"]
    assert resource_attributes["model_name"] == "test-model"
    assert resource_attributes["rank"] == 5
    assert resource_attributes["dp_rank"] == 2

    metrics = {
        metric["name"]: metric["data"]["data_points"][0]
        for metric in resource_metrics["scope_metrics"][0]["metrics"]
    }
    expected_values = {
        "vllm.generation_tokens": 12,
        "vllm.prompt_tokens": 100,
        "vllm.cached_prompt_tokens": 30,
        "vllm.preempted_requests": 1,
        "vllm.finished_requests": 1,
        "vllm.prefix_cache_queries": 10,
        "vllm.prefix_cache_hits": 7,
        "vllm.kv_cache_usage": 0.5,
        "vllm.num_running_requests": 4,
        "vllm.num_waiting_requests": 2,
    }
    expected_histograms = {
        "vllm.time_to_first_token": (1, 12.0),
        "vllm.inter_token_latency": (2, 30.0),
        "vllm.decode_time": (1, 30.0),
        "vllm.queue_time": (1, 5.0),
        "vllm.e2e_latency": (1, 50.0),
    }
    assert set(metrics) == expected_values.keys() | expected_histograms.keys()
    for name, expected_value in expected_values.items():
        assert metrics[name]["value"] == pytest.approx(expected_value)
    for name, (expected_count, expected_sum) in expected_histograms.items():
        assert metrics[name]["count"] == expected_count
        assert metrics[name]["sum"] == pytest.approx(expected_sum)


def test_record_disables_on_instrument_failure(no_otel_env, caplog):
    log = VllmOtelStatLogger.Config().build(
        vllm_config=_vllm_config(), context=_context()
    )

    class FailingCounter:
        def add(self, _value):
            raise ValueError("simulated metrics failure")

    log._c_gen_tokens = FailingCounter()
    log._enabled = True

    with caplog.at_level(logging.WARNING):
        log.record(None, None)

    assert log._enabled is False
    assert "record() failed" in caplog.text


def test_record_disables_on_extract_failure(no_otel_env):
    log = VllmOtelStatLogger.Config().build(
        vllm_config=_vllm_config(), context=_context()
    )
    log._enabled = True

    log.record(None, SimpleNamespace())

    assert log._enabled is False
