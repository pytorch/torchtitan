# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for `torchtitan.experiments.rl.observability.metrics`."""

from __future__ import annotations

import logging
import math
import os
from unittest.mock import MagicMock

import pytest

from torchtitan.components.metrics import BaseLogger, WandBLogger
from torchtitan.experiments.rl.observability import metrics as m


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


class TestMean:
    def test_from_list_basic(self) -> None:
        agg = m.aggregate_metrics([m.Metric("k", m.Mean.from_list([1.0, 2.0, 3.0]))])
        assert agg == {"k/mean": 2.0}

    def test_weighted_combine(self) -> None:
        # Two `Mean` records under the same key combine as
        # `(10 + 20) / (4 + 1) = 6.0`.
        agg = m.aggregate_metrics(
            [
                m.Metric("k", m.Mean(10.0, count=4)),
                m.Metric("k", m.Mean(20.0, count=1)),
            ]
        )
        assert agg == {"k/mean": 6.0}

    def test_empty_list_filtered(self) -> None:
        agg = m.aggregate_metrics([m.Metric("k", m.Mean.from_list([]))])
        assert agg == {}

    def test_single_value(self) -> None:
        agg = m.aggregate_metrics([m.Metric("k", m.Mean(5.0))])
        assert agg == {"k/mean": 5.0}


class TestMaxMin:
    def test_max_from_list(self) -> None:
        agg = m.aggregate_metrics([m.Metric("k", m.Max.from_list([1.0, 5.0, 3.0]))])
        assert agg == {"k/max": 5.0}

    def test_min_from_list(self) -> None:
        agg = m.aggregate_metrics([m.Metric("k", m.Min.from_list([1.0, 5.0, 3.0]))])
        assert agg == {"k/min": 1.0}

    def test_max_empty_filtered(self) -> None:
        agg = m.aggregate_metrics([m.Metric("k", m.Max.from_list([]))])
        assert agg == {}

    def test_min_empty_filtered(self) -> None:
        agg = m.aggregate_metrics([m.Metric("k", m.Min.from_list([]))])
        assert agg == {}

    def test_max_single(self) -> None:
        agg = m.aggregate_metrics([m.Metric("k", m.Max(7.0))])
        assert agg == {"k/max": 7.0}

    def test_max_combine(self) -> None:
        agg = m.aggregate_metrics(
            [m.Metric("k", m.Max(1.0)), m.Metric("k", m.Max(9.0))]
        )
        assert agg == {"k/max": 9.0}


class TestStd:
    def test_single_value_zero(self) -> None:
        agg = m.aggregate_metrics([m.Metric("k", m.Std(5.0))])
        assert agg["k/std"] == pytest.approx(0.0, abs=1e-9)

    def test_from_list_matches_population_std(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        # Population std (ddof=0) of [1,2,3,4] = sqrt(1.25)
        expected = math.sqrt(1.25)
        agg = m.aggregate_metrics([m.Metric("k", m.Std.from_list(values))])
        assert agg["k/std"] == pytest.approx(expected, abs=1e-7)

    def test_combine_equals_concatenation(self) -> None:
        # Combining `Std.from_list([1,2])` and `Std.from_list([3,4])`
        # equals `Std.from_list([1,2,3,4])` within FP tolerance.
        a = m.Std.from_list([1.0, 2.0])
        b = m.Std.from_list([3.0, 4.0])
        combined = m.aggregate_metrics([m.Metric("k", a), m.Metric("k", b)])
        full = m.aggregate_metrics(
            [m.Metric("k", m.Std.from_list([1.0, 2.0, 3.0, 4.0]))]
        )
        assert combined["k/std"] == pytest.approx(full["k/std"], abs=1e-9)

    def test_empty_filtered(self) -> None:
        agg = m.aggregate_metrics([m.Metric("k", m.Std.from_list([]))])
        assert agg == {}


class TestStats:
    def test_five_keys_emitted(self) -> None:
        agg = m.aggregate_metrics(
            [m.Metric("k", m.Stats.from_list([1.0, 2.0, 3.0, 4.0]))]
        )
        assert set(agg) == {
            "k/_max",
            "k/_mean",
            "k/_min",
            "k/_std",
            "k/_sum",
        }
        assert agg["k/_sum"] == 10.0
        assert agg["k/_mean"] == pytest.approx(2.5)
        assert agg["k/_min"] == 1.0
        assert agg["k/_max"] == 4.0
        assert agg["k/_std"] == pytest.approx(math.sqrt(1.25), abs=1e-7)

    def test_empty_keeps_sum_zero(self) -> None:
        # `Stats.from_list([])` becomes a zero-count record. The
        # aggregator filters NaN keys (mean/min/max/std) but keeps
        # `_sum=0` since it isn't NaN.
        agg = m.aggregate_metrics([m.Metric("k", m.Stats.from_list([]))])
        assert agg == {"k/_sum": 0.0}

    def test_combine_two_records(self) -> None:
        a = m.Stats.from_list([1.0, 2.0])
        b = m.Stats.from_list([3.0, 4.0])
        combined = m.aggregate_metrics([m.Metric("k", a), m.Metric("k", b)])
        assert combined["k/_sum"] == 10.0
        assert combined["k/_mean"] == pytest.approx(2.5)
        assert combined["k/_min"] == 1.0
        assert combined["k/_max"] == 4.0


class TestNoReduce:
    def test_pass_through(self) -> None:
        agg = m.aggregate_metrics([m.Metric("loss", m.NoReduce(0.5))])
        assert agg == {"loss": 0.5}

    def test_from_list_single(self) -> None:
        agg = m.aggregate_metrics([m.Metric("k", m.NoReduce.from_list([3.0]))])
        assert agg == {"k": 3.0}

    def test_from_list_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            m.NoReduce.from_list([])

    def test_from_list_multiple_raises(self) -> None:
        with pytest.raises(ValueError):
            m.NoReduce.from_list([1.0, 2.0])

    def test_two_entries_same_key_raises(self) -> None:
        with pytest.raises(ValueError):
            m.aggregate_metrics(
                [m.Metric("k", m.NoReduce(1.0)), m.Metric("k", m.NoReduce(2.0))]
            )


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


class TestAggregator:
    def test_groups_by_key_and_reduction(self) -> None:
        # Two reductions under the same key with distinct suffixes
        # both appear.
        agg = m.aggregate_metrics(
            [
                m.Metric("len", m.Mean.from_list([1.0, 2.0, 3.0])),
                m.Metric("len", m.Max.from_list([1.0, 2.0, 3.0])),
            ]
        )
        assert agg == {"len/mean": 2.0, "len/max": 3.0}

    def test_nan_filtered(self) -> None:
        agg = m.aggregate_metrics(
            [
                m.Metric("a", m.Max.from_list([])),  # NaN — filtered
                m.Metric("b", m.Mean(2.0)),
            ]
        )
        assert agg == {"b/mean": 2.0}

    def test_duplicate_no_reduce_raises(self) -> None:
        with pytest.raises(ValueError):
            m.aggregate_metrics(
                [
                    m.Metric("loss", m.NoReduce(1.0)),
                    m.Metric("loss", m.NoReduce(2.0)),
                ]
            )

    def test_empty(self) -> None:
        assert m.aggregate_metrics([]) == {}

    def test_mean_and_stats_no_collision(self) -> None:
        # `Mean(v)` writes `len/mean`; `Stats(v)` writes `len/_mean`.
        # Both can appear in the same step.
        agg = m.aggregate_metrics(
            [m.Metric("len", m.Mean(2.0)), m.Metric("len", m.Stats(2.0))]
        )
        assert "len/mean" in agg
        assert "len/_mean" in agg
        assert agg["len/mean"] == 2.0
        assert agg["len/_mean"] == 2.0


# ---------------------------------------------------------------------------
# log_to_console (stateless utility)
# ---------------------------------------------------------------------------


class TestLogToConsole:
    def test_alphabetical_when_no_allow_list(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO):
            m.log_to_console(7, {"b": 2.0, "a": 1.0}, allow_list=None)
        records = [r for r in caplog.records if "step:" in r.getMessage()]
        assert len(records) == 1
        msg = records[0].getMessage()
        assert "step:  7" in msg
        # Alphabetical column order — `a` before `b`.
        assert msg.index("a: 1.0") < msg.index("b: 2.0")

    def test_allow_list_none_prints_all(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO):
            m.log_to_console(
                0,
                {"loss/total": 0.5, "rollout/reward/_mean": 1.0},
                allow_list=None,
            )
        msgs = [r.getMessage() for r in caplog.records if "step:" in r.getMessage()]
        assert msgs, "expected one log line"
        msg = msgs[-1]
        assert "loss/total" in msg and "rollout/reward/_mean" in msg

    def test_allow_list_empty_prints_nothing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO):
            m.log_to_console(0, {"loss/total": 0.5}, allow_list=[])
        assert not any("step:" in r.getMessage() for r in caplog.records)

    def test_allow_list_regex_search(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO):
            m.log_to_console(
                0,
                {"some_loss_metric": 1.0, "other_metric": 2.0},
                allow_list=["loss"],
            )
        msgs = [r.getMessage() for r in caplog.records if "step:" in r.getMessage()]
        assert msgs, "expected one log line"
        msg = msgs[-1]
        assert "some_loss_metric: 1.0" in msg
        assert "other_metric" not in msg

    def test_pattern_order_preserved(self, caplog: pytest.LogCaptureFixture) -> None:
        """Column order follows ``allow_list`` pattern order (not
        alphabetical), so callers can pin the layout per context."""
        with caplog.at_level(logging.INFO):
            m.log_to_console(
                0,
                {"foo": 1.0, "bar": 2.0, "baz": 3.0},
                allow_list=[r"^baz$", r"^foo$", r"^bar$"],
            )
        msgs = [r.getMessage() for r in caplog.records if "step:" in r.getMessage()]
        assert msgs
        msg = msgs[-1]
        assert msg.index("baz: 3.0") < msg.index("foo: 1.0") < msg.index("bar: 2.0")

    def test_no_match_no_print(self, caplog: pytest.LogCaptureFixture) -> None:
        """Allow-list matches no key in this step => no console output.
        Lets train/validation share one MetricLogger.log call with
        different allow lists; the off-context call stays silent."""
        with caplog.at_level(logging.INFO):
            m.log_to_console(
                0,
                {"loss/total": 0.5},
                allow_list=[r"^validation/reward/_mean$"],
            )
        assert not any("step:" in r.getMessage() for r in caplog.records)

    def test_prefix_renders_before_step(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO):
            m.log_to_console(
                0,
                {"validation/reward": 0.9},
                allow_list=None,
                prefix="validate ",
            )
        msgs = [r.getMessage() for r in caplog.records if "step:" in r.getMessage()]
        assert msgs
        assert "validate step:  0" in msgs[-1]


@pytest.mark.parametrize(
    "value, expected",
    [
        (3.67060, "3.67"),
        (0.00123, "0.0012"),
        (1234.5, "1234.5"),
        (True, "True"),
        (0, "0"),
        (0.0, "0"),
        (-0.001234, "-0.0012"),
        (99.999, "100.00"),
        (5, "5.0"),
        # Very small values fall back to scientific so train/lr ≈ 2e-6
        # doesn't render as `0.00000`.
        (2e-6, "2.0e-06"),
        (1e-5, "0.00001"),  # boundary: 5 decimals still resolves the digit
        (1e-7, "1.0e-07"),
        (-3.5e-8, "-3.5e-08"),
    ],
)
def test_fmt_value(value, expected) -> None:
    assert m._fmt_value(value) == expected


# ---------------------------------------------------------------------------
# Backend identity
# ---------------------------------------------------------------------------


def test_wandb_metric_logger_is_wandb_logger() -> None:
    assert m.WandbMetricLogger is WandBLogger


def test_metric_backend_is_base_logger() -> None:
    assert m.MetricBackend is BaseLogger


# ---------------------------------------------------------------------------
# MetricLogger fan-out + build
# ---------------------------------------------------------------------------


class _RecordingBackend(m.MetricBackend):
    def __init__(self) -> None:
        self.calls: list[tuple[dict, int]] = []
        self.closed = False

    def log(self, metrics: dict, step: int) -> None:
        self.calls.append((dict(metrics), step))

    def close(self) -> None:
        self.closed = True


class _RaisingBackend(m.MetricBackend):
    def __init__(self) -> None:
        self.log_calls = 0
        self.close_calls = 0

    def log(self, metrics: dict, step: int) -> None:
        self.log_calls += 1
        raise RuntimeError("boom")

    def close(self) -> None:
        self.close_calls += 1
        raise RuntimeError("boom on close")


class TestMetricLoggerFanOut:
    def test_log_aggregates_and_dispatches(self) -> None:
        b1 = _RecordingBackend()
        b2 = _RecordingBackend()
        logger_inst = m.MetricLogger([b1, b2])
        logger_inst.log(
            5,
            [
                m.Metric("k", m.Mean.from_list([1.0, 2.0, 3.0])),
                m.Metric("k", m.Max.from_list([1.0, 2.0, 3.0])),
            ],
        )
        assert b1.calls == [({"k/mean": 2.0, "k/max": 3.0}, 5)]
        assert b2.calls == [({"k/mean": 2.0, "k/max": 3.0}, 5)]

    def test_log_console_allow_list_per_call(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Per-call allow list controls only the console line.
        Non-matching list keeps the line silent; matching list renders.
        Backends always see every call regardless of console arg."""
        backend = _RecordingBackend()
        logger_inst = m.MetricLogger([backend])

        with caplog.at_level(logging.INFO):
            # 1) explicit empty => silent
            logger_inst.log(
                0,
                [m.Metric("loss/total", m.NoReduce(1.0))],
                console_allow_list=[],
            )
            # 2) non-matching pattern => silent (off-context)
            logger_inst.log(
                1,
                [m.Metric("loss/total", m.NoReduce(2.0))],
                console_allow_list=[r"^validation"],
            )
            # 3) matching pattern => one line
            logger_inst.log(
                2,
                [m.Metric("loss/total", m.NoReduce(3.0))],
                console_allow_list=[r"^loss"],
            )

        msgs = [r.getMessage() for r in caplog.records if "step:" in r.getMessage()]
        assert len(msgs) == 1
        assert "step:  2" in msgs[0]
        # Backend always sees every step regardless of console arg.
        assert [step for _, step in backend.calls] == [0, 1, 2]

    def test_log_isolates_failures(self, caplog: pytest.LogCaptureFixture) -> None:
        good = _RecordingBackend()
        bad = _RaisingBackend()
        # Both orderings: ensure the good backend always runs.
        for backends in ([bad, good], [good, bad]):
            good.calls.clear()
            bad.log_calls = 0
            logger_inst = m.MetricLogger(backends)
            with caplog.at_level(logging.ERROR, logger="torchtitan"):
                logger_inst.log(0, [m.Metric("k", m.NoReduce(1.0))])
            assert bad.log_calls == 1
            assert good.calls == [({"k": 1.0}, 0)]

    def test_close_isolates_failures(self, caplog: pytest.LogCaptureFixture) -> None:
        good = _RecordingBackend()
        bad = _RaisingBackend()
        logger_inst = m.MetricLogger([bad, good])
        with caplog.at_level(logging.ERROR, logger="torchtitan"):
            logger_inst.close()
        assert bad.close_calls == 1
        assert good.closed is True


class TestMetricLoggerBuild:
    def test_default_builds_no_backends(self) -> None:
        """Console isn't a backend anymore; default config builds an
        empty backend list. ``log()`` with ``console_allow_list=[]``
        suppresses the console line and is a no-op."""
        logger_inst = m.MetricLogger.build(m.MetricsConfig())
        assert logger_inst._backends == []
        logger_inst.log(0, [m.Metric("k", m.NoReduce(1.0))], console_allow_list=[])
        logger_inst.close()

    def test_wandb_without_log_dir_raises(self) -> None:
        with pytest.raises(ValueError, match="log_dir is required"):
            m.MetricLogger.build(m.MetricsConfig(enable_wandb=True))

    def test_wandb_with_log_dir(self, tmp_path, monkeypatch) -> None:
        import wandb  # noqa: F401  (ensures real module exists at import time)

        captured = {}
        fake_wandb = MagicMock()
        fake_wandb.init = MagicMock(
            side_effect=lambda **kwargs: captured.update(kwargs)
        )
        # `WandBLogger` does `import wandb` inside `__init__`, so swap the
        # cached entry in `sys.modules`.
        monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

        config_dict = {"job": "rl-debug"}
        log_dir = str(tmp_path / "wandb_dir")

        logger_inst = m.MetricLogger.build(
            m.MetricsConfig(enable_wandb=True),
            log_dir=log_dir,
            config_dict=config_dict,
        )
        assert len(logger_inst._backends) == 1
        backend = logger_inst._backends[0]
        assert isinstance(backend, m.WandbMetricLogger)
        # `wandb.init` was called with our log_dir + config_dict.
        assert captured.get("dir") == log_dir
        assert captured.get("config") == config_dict

        # `log` forwards to the stub.
        logger_inst.log(3, [m.Metric("k", m.NoReduce(1.0))])
        fake_wandb.log.assert_called_once_with({"k": 1.0}, step=3)

        # `close` calls `wandb.finish` when there's a run.
        fake_wandb.run = MagicMock()
        logger_inst.close()
        fake_wandb.finish.assert_called_once()

    def test_train_validation_allow_lists_round_trip(self) -> None:
        """``MetricsConfig`` carries the two per-context allow lists; the
        controller picks them up at the callsite. The build method
        ignores them — they aren't part of any backend."""
        cfg = m.MetricsConfig(
            train_console_allow_list=[r"^loss/total$"],
            validation_console_allow_list=[r"^validation/reward$"],
        )
        # Build should succeed and produce no backends (no W&B configured).
        logger_inst = m.MetricLogger.build(cfg)
        assert logger_inst._backends == []
        # Round-trip: caller can read them straight off the config.
        assert cfg.train_console_allow_list == [r"^loss/total$"]
        assert cfg.validation_console_allow_list == [r"^validation/reward$"]

    def test_wandb_project_default_titan_rl(self, tmp_path, monkeypatch) -> None:
        """When ``WANDB_PROJECT`` is unset, ``MetricLogger.build`` should
        default it to ``titan_rl`` so RL runs land in the dedicated W&B
        project instead of the generic ``torchtitan`` one."""
        monkeypatch.delenv("WANDB_PROJECT", raising=False)

        captured = {}
        fake_wandb = MagicMock()
        fake_wandb.init = MagicMock(
            side_effect=lambda **kwargs: captured.update(
                {**kwargs, "_env_project": os.environ.get("WANDB_PROJECT")}
            )
        )
        monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

        m.MetricLogger.build(
            m.MetricsConfig(enable_wandb=True),
            log_dir=str(tmp_path),
        )
        assert captured.get("_env_project") == "titan_rl"

    def test_wandb_project_env_var_wins(self, tmp_path, monkeypatch) -> None:
        """User-set ``WANDB_PROJECT`` takes precedence over the
        ``titan_rl`` default."""
        monkeypatch.setenv("WANDB_PROJECT", "my-other-project")

        captured = {}
        fake_wandb = MagicMock()
        fake_wandb.init = MagicMock(
            side_effect=lambda **kwargs: captured.update(
                {**kwargs, "_env_project": os.environ.get("WANDB_PROJECT")}
            )
        )
        monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

        m.MetricLogger.build(
            m.MetricsConfig(enable_wandb=True),
            log_dir=str(tmp_path),
        )
        assert captured.get("_env_project") == "my-other-project"
