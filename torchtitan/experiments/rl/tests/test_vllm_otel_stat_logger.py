# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
from types import SimpleNamespace

import pytest

from torchtitan.experiments.rl.observability.vllm_otel_stat_logger import (
    VllmOtelStatLogger,
)
from torchtitan.experiments.rl.observability.vllm_stat_common import StatLoggerContext


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


def test_inert_on_dp_head_without_endpoint(no_otel_env):
    log = VllmOtelStatLogger(
        _vllm_config(), 0, context=_context(rank=0, tp_rank=0, dp_rank=0)
    )
    assert log._should_log is True  # tp_rank==0
    assert log._enabled is False  # ...but no endpoint -> inert
    # No-ops that must never raise and never touch (uncreated) instruments.
    log.record(None, None)
    log.log()


def test_disabled_off_dp_head(no_otel_env):
    log = VllmOtelStatLogger(
        _vllm_config(tp_size=2),
        0,
        context=_context(rank=1, tp_rank=1, dp_rank=0),
    )
    assert log._should_log is False
    assert log._enabled is False
    log.record(None, None)
    log.log()


def test_exporter_none_is_inert(monkeypatch):
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", "none")
    log = VllmOtelStatLogger(
        _vllm_config(), 0, context=_context(rank=0, tp_rank=0, dp_rank=0)
    )
    assert log._enabled is False


def test_endpoint_without_exporter_is_inert(no_otel_env, monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")

    log = VllmOtelStatLogger(_vllm_config(), context=_context())

    assert log._enabled is False


def test_sdk_disabled_is_inert(no_otel_env, monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", "jsonl")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")

    with caplog.at_level(logging.WARNING):
        log = VllmOtelStatLogger(
            _vllm_config(), context=_context(output_dir=str(tmp_path))
        )

    assert log._enabled is False
    assert "OTEL_SDK_DISABLED is set to true" in caplog.text
    assert not (tmp_path / "vllm_metrics").exists()


@pytest.mark.parametrize("exporter", ["console", "jsonl,otlp", "unknown"])
def test_unsupported_exporter_is_inert(no_otel_env, monkeypatch, caplog, exporter):
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", exporter)

    with caplog.at_level(logging.WARNING):
        log = VllmOtelStatLogger(_vllm_config(), context=_context())

    assert log._enabled is False
    assert "unsupported OTEL_METRICS_EXPORTER" in caplog.text


def test_jsonl_requires_output_dir(no_otel_env, monkeypatch, caplog):
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", "jsonl")

    with caplog.at_level(logging.WARNING):
        log = VllmOtelStatLogger(_vllm_config(), context=_context(output_dir=""))

    assert log._enabled is False
    assert "requires an output directory" in caplog.text


def test_otlp_requires_http_protobuf(no_otel_env, monkeypatch, caplog):
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", "otlp")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")

    with caplog.at_level(logging.WARNING):
        log = VllmOtelStatLogger(_vllm_config(), context=_context())

    assert log._enabled is False
    assert "uses the OTLP HTTP/protobuf exporter" in caplog.text


def test_jsonl_export_is_asynchronous(no_otel_env, monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", "jsonl")
    monkeypatch.setenv("OTEL_METRIC_EXPORT_TIMEOUT", "1234")
    monkeypatch.setenv("VLLM_LOG_STATS_INTERVAL", "600")
    log = VllmOtelStatLogger(_vllm_config(), context=_context(output_dir=str(tmp_path)))

    assert log._enabled is True
    reader = log._provider._metric_readers[0]
    assert reader._daemon_thread is not None
    assert reader._export_interval_millis == 600000
    assert reader._export_timeout_millis == 1234

    with caplog.at_level(logging.INFO):
        log.log_engine_initialized()
    assert "VLLM_LOG_STATS_INTERVAL=600" in caplog.text

    # vLLM calls log() from LLMEngine.step(). It must remain a no-op instead of
    # performing synchronous exporter I/O on the generation path.
    monkeypatch.setattr(
        log._provider,
        "force_flush",
        lambda *args, **kwargs: pytest.fail("force_flush called from log()"),
    )
    log.log()
    log._provider.shutdown(timeout_millis=1000)
