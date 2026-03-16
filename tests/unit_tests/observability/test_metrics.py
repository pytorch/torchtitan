# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for metrics.py: MetricValue types, record_metric, ExperimentJSONFormatter."""

import json
import logging

import pytest
from torchtitan.observability import step_state

from torchtitan.observability._constants import (
    EXPERIMENT_LOGGER_NAME,
    SYSTEM_LOGGER_NAME,
)
from torchtitan.observability.metrics import (
    ExperimentJSONFormatter,
    MaxMetric,
    MeanMetric,
    MinMetric,
    NoOpMetric,
    record_metric,
    REDUCE_REGISTRY,
    SumMetric,
)
from torchtitan.observability.step_state import set_step
from torchtitan.observability.structured_logging import init_observability


@pytest.fixture(autouse=True)
def reset_step():
    step_state._STEP = None
    yield
    step_state._STEP = None


@pytest.fixture
def exp_logger():
    """Provide clean experiment + system loggers for testing."""
    exp_log = logging.getLogger(EXPERIMENT_LOGGER_NAME)
    sys_log = logging.getLogger(SYSTEM_LOGGER_NAME)
    exp_orig = (exp_log.handlers[:], exp_log.level, exp_log.propagate)
    sys_orig = (sys_log.handlers[:], sys_log.level, sys_log.propagate)
    yield exp_log
    exp_log.handlers, exp_log.level, exp_log.propagate = exp_orig
    sys_log.handlers, sys_log.level, sys_log.propagate = sys_orig


# ---------------------------------------------------------------------------
# MetricValue types
# ---------------------------------------------------------------------------


class TestMeanMetric:
    def test_from_value(self):
        m = MeanMetric(sum=2.5)
        state = m.get_state()
        assert state["reduce"] == "MeanMetric"
        assert state["sum"] == 2.5
        assert state["weight"] == 1.0

    def test_from_sum_weight(self):
        m = MeanMetric(sum=10.0, weight=4.0)
        state = m.get_state()
        assert state["sum"] == 10.0
        assert state["weight"] == 4.0

    def test_reduce(self):
        states = [
            {"sum": 6.0, "weight": 3.0},
            {"sum": 4.0, "weight": 2.0},
        ]
        result = MeanMetric.get_reduced_value_from_states(states)
        assert result == pytest.approx(2.0)  # 10/5

    def test_default_weight(self):
        m = MeanMetric(sum=5.0)
        assert m._weight == 1.0


class TestMaxMetric:
    def test_basic(self):
        assert MaxMetric(5.0).get_state()["value"] == 5.0

    def test_reduce(self):
        states = [{"value": 3.0}, {"value": 7.0}, {"value": 1.0}]
        assert MaxMetric.get_reduced_value_from_states(states) == 7.0


class TestMinMetric:
    def test_reduce(self):
        states = [{"value": 3.0}, {"value": 7.0}, {"value": 1.0}]
        assert MinMetric.get_reduced_value_from_states(states) == 1.0


class TestSumMetric:
    def test_reduce(self):
        states = [{"value": 3.0}, {"value": 7.0}, {"value": 1.0}]
        assert SumMetric.get_reduced_value_from_states(states) == 11.0


class TestNoOpMetric:
    def test_basic(self):
        m = NoOpMetric(value=0.038)
        state = m.get_state()
        assert state["reduce"] == "NoOpMetric"
        assert state["value"] == 0.038

    def test_reduce_takes_first(self):
        states = [{"value": 0.5}, {"value": 0.6}, {"value": 0.7}]
        assert NoOpMetric.get_reduced_value_from_states(states) == 0.5


class TestReduceRegistry:
    def test_all_types_registered(self):
        assert "MeanMetric" in REDUCE_REGISTRY
        assert "MaxMetric" in REDUCE_REGISTRY
        assert "MinMetric" in REDUCE_REGISTRY
        assert "SumMetric" in REDUCE_REGISTRY
        assert "NoOpMetric" in REDUCE_REGISTRY


# ---------------------------------------------------------------------------
# ExperimentJSONFormatter
# ---------------------------------------------------------------------------


class TestExperimentJSONFormatter:
    def test_formats_metric_entry(self):
        fmt = ExperimentJSONFormatter(rank=0, source="trainer")
        set_step(5)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="experiment_metric",
            args=None,
            exc_info=None,
        )
        record._metric_entry = {
            "key": "reward",
            "reduce": "MaxMetric",
            "value": 0.95,
        }
        output = fmt.format(record)
        parsed = json.loads(output)
        assert parsed["key"] == "reward"
        assert parsed["step"] == 5
        assert parsed["rank"] == 0
        assert parsed["source"] == "trainer"
        assert "caller" in parsed
        assert "timestamp" in parsed

    def test_raises_without_step(self):
        fmt = ExperimentJSONFormatter(rank=0, source="trainer")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=None,
            exc_info=None,
        )
        record._metric_entry = {"key": "x", "reduce": "MaxMetric", "value": 1.0}
        with pytest.raises(ValueError, match="No step"):
            fmt.format(record)


# ---------------------------------------------------------------------------
# record_metric end-to-end
# ---------------------------------------------------------------------------


class TestRecordMetricEndToEnd:
    def test_writes_to_experiment_jsonl(self, tmp_path, exp_logger):
        init_observability(rank=0, source="trainer", output_dir=str(tmp_path))
        set_step(42)

        record_metric("reward", MaxMetric(0.95))
        record_metric("lr", MeanMetric(sum=1e-4))

        for h in exp_logger.handlers:
            h.flush()

        exp_dir = tmp_path / "experiment_logs"
        jsonl_files = list(exp_dir.glob("*.jsonl"))
        assert len(jsonl_files) == 1

        with open(jsonl_files[0]) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 2
        keys = {line["key"] for line in lines}
        assert keys == {"reward", "lr"}

    def test_raises_without_step(self):
        with pytest.raises(ValueError, match="No step"):
            record_metric("x", MaxMetric(1.0))
