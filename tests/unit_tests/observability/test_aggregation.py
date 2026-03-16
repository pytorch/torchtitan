# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for aggregation.py: aggregate() and logging_worker."""

import multiprocessing

import pytest

from torchtitan.observability.aggregation import aggregate, logging_worker


class TestAggregate:
    def test_empty(self):
        assert aggregate([]) == {}

    def test_mean_metric(self):
        entries = [
            {"key": "tps", "reduce": "MeanMetric", "sum": 1000, "weight": 1},
            {"key": "tps", "reduce": "MeanMetric", "sum": 1200, "weight": 1},
        ]
        result = aggregate(entries)
        assert result["tps"] == pytest.approx(1100.0)

    def test_max_metric(self):
        entries = [
            {"key": "mem", "reduce": "MaxMetric", "value": 14.0},
            {"key": "mem", "reduce": "MaxMetric", "value": 15.5},
        ]
        result = aggregate(entries)
        assert result["mem"] == pytest.approx(15.5)

    def test_noop_metric(self):
        entries = [
            {"key": "loss", "reduce": "NoOpMetric", "value": 0.5},
            {"key": "loss", "reduce": "NoOpMetric", "value": 0.6},
        ]
        result = aggregate(entries)
        assert result["loss"] == pytest.approx(0.5)  # takes first

    def test_mixed_keys(self):
        entries = [
            {"key": "loss", "reduce": "NoOpMetric", "value": 0.5},
            {"key": "tps", "reduce": "MeanMetric", "sum": 1000, "weight": 1},
            {"key": "mem", "reduce": "MaxMetric", "value": 14.0},
        ]
        result = aggregate(entries)
        assert len(result) == 3
        assert result["loss"] == 0.5
        assert result["tps"] == 1000.0
        assert result["mem"] == 14.0


class TestLoggingWorkerBasic:
    def test_reads_jsonl_and_shuts_down(self, tmp_path):
        """Logging worker reads JSONL, aggregates, and shuts down cleanly."""
        log_dir = tmp_path / "experiment_logs"
        log_dir.mkdir()

        fp = log_dir / "rank_0.jsonl"
        fp.write_text(
            '{"key": "loss", "reduce": "NoOpMetric", "value": 0.5, "step": 1}\n'
            '{"key": "tps", "reduce": "MeanMetric", "sum": 1000, "weight": 1, "step": 1}\n'
        )

        queue = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=logging_worker,
            args=(queue, str(tmp_path)),
            kwargs={"queue_timeout_s": 5},
        )
        p.start()
        queue.put((1, False))
        queue.put(None)  # shutdown
        p.join(timeout=10)
        assert p.exitcode == 0

    def test_skip_historical_data(self, tmp_path):
        """On startup, worker skips existing JSONL data (checkpoint resume)."""
        log_dir = tmp_path / "experiment_logs"
        log_dir.mkdir()

        fp = log_dir / "rank_0.jsonl"
        # Write old data before worker starts
        fp.write_text(
            '{"key": "loss", "reduce": "NoOpMetric", "value": 99.0, "step": 50}\n'
        )

        queue = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=logging_worker,
            args=(queue, str(tmp_path)),
            kwargs={"queue_timeout_s": 5},
        )
        p.start()

        # Write new data after worker started
        with open(fp, "a") as f:
            f.write(
                '{"key": "loss", "reduce": "NoOpMetric", "value": 0.5, "step": 51}\n'
            )

        queue.put((51, False))
        queue.put(None)
        p.join(timeout=10)
        assert p.exitcode == 0

    def test_multi_rank_aggregation(self, tmp_path):
        """Worker aggregates metrics from multiple rank files."""
        log_dir = tmp_path / "experiment_logs"
        log_dir.mkdir()

        for rank in range(4):
            fp = log_dir / f"rank_{rank}.jsonl"
            tps = 1000 + rank * 100
            fp.write_text(
                f'{{"key": "tps", "reduce": "MeanMetric", "sum": {tps}, "weight": 1, "step": 1}}\n'
            )

        queue = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=logging_worker,
            args=(queue, str(tmp_path)),
            kwargs={"queue_timeout_s": 5},
        )
        p.start()
        queue.put((1, False))
        queue.put(None)
        p.join(timeout=10)
        assert p.exitcode == 0
