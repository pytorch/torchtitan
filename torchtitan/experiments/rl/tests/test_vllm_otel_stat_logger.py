# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from types import SimpleNamespace

import pytest

from torchtitan.experiments.rl.observability.vllm_otel_stat_logger import (
    VllmOtelStatLogger,
)
from torchtitan.experiments.rl.observability.vllm_stat_common import StatLoggerContext

_OTEL_ENV = (
    "OTEL_METRICS_EXPORTER",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
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
    for key in _OTEL_ENV:
        monkeypatch.delenv(key, raising=False)


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
    for key in _OTEL_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", "none")
    log = VllmOtelStatLogger(
        _vllm_config(), 0, context=_context(rank=0, tp_rank=0, dp_rank=0)
    )
    assert log._enabled is False
