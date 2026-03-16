# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Integration tests for experiment JSONL from toy_spmd.

Verifies record_metric output. Run toy_spmd first to generate output files.
"""

import json
import os
from glob import glob

import pytest

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs", "toy_spmd"
)


@pytest.fixture
def experiment_logs_dir():
    """Path to experiment_logs from a prior toy_spmd run."""
    path = os.path.join(OUTPUT_DIR, "experiment_logs")
    if not os.path.isdir(path):
        pytest.fail(
            f"No experiment_logs at {path}. Run toy_spmd first to generate outputs."
        )
    return path


class TestExperimentJSONL:
    def test_experiment_jsonl_files_exist(self, experiment_logs_dir):
        files = glob(os.path.join(experiment_logs_dir, "*.jsonl"))
        assert len(files) >= 1

    def test_experiment_jsonl_has_expected_keys(self, experiment_logs_dir):
        files = glob(os.path.join(experiment_logs_dir, "*.jsonl"))
        keys = set()
        for fp in files:
            with open(fp) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    keys.add(record.get("key"))

        assert "training/loss_mean" in keys, f"Missing loss metric. Found: {keys}"
        assert "training/grad_norm_max" in keys, f"Missing grad_norm. Found: {keys}"
        assert "training/lr" in keys, f"Missing lr. Found: {keys}"

    def test_experiment_jsonl_has_required_fields(self, experiment_logs_dir):
        files = glob(os.path.join(experiment_logs_dir, "*.jsonl"))
        for fp in files:
            with open(fp) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    assert "key" in record
                    assert "reduce" in record
                    assert "step" in record
                    assert "rank" in record
                    assert "caller" in record
                    assert "timestamp" in record
                    return  # One record is enough

    def test_experiment_jsonl_reduce_types_correct(self, experiment_logs_dir):
        files = glob(os.path.join(experiment_logs_dir, "*.jsonl"))
        reduce_by_key: dict[str, str] = {}
        for fp in files:
            with open(fp) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    reduce_by_key[record["key"]] = record["reduce"]

        assert reduce_by_key.get("training/loss_mean") == "NoOpMetric"
        assert reduce_by_key.get("training/grad_norm_max") == "MaxMetric"
        assert reduce_by_key.get("training/lr") == "NoOpMetric"
