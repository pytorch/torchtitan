# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Integration tests for toy_rl output.

Run toy_rl first to generate output files.
"""

import json
import os
from glob import glob

import pytest

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs", "toy_rl"
)


@pytest.fixture
def experiment_logs_dir():
    path = os.path.join(OUTPUT_DIR, "experiment_logs")
    if not os.path.isdir(path):
        pytest.fail(
            f"No experiment_logs at {path}. Run toy_rl first to generate outputs."
        )
    return path


@pytest.fixture
def rollout_dir():
    path = os.path.join(OUTPUT_DIR, "rollouts")
    if not os.path.isdir(path):
        pytest.fail(
            f"No rollouts at {path}. Run toy_rl first to generate outputs."
        )
    return path


class TestRLExperimentJSONL:
    def test_experiment_jsonl_has_training_metrics(self, experiment_logs_dir):
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

        assert "training/loss_mean" in keys, f"Missing loss. Found: {keys}"
        assert "training/grad_norm_max" in keys, f"Missing grad_norm. Found: {keys}"
        assert "training/lr" in keys, f"Missing lr. Found: {keys}"
        assert "trainer_throughput/tps_mean" in keys, f"Missing tps. Found: {keys}"


class TestRolloutLogger:
    def test_rollout_file_exists(self, rollout_dir):
        files = glob(os.path.join(rollout_dir, "*.jsonl"))
        assert len(files) >= 1, f"No rollout JSONL files in {rollout_dir}"

    def test_rollout_has_expected_fields(self, rollout_dir):
        files = glob(os.path.join(rollout_dir, "*.jsonl"))
        for fp in files:
            with open(fp) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    assert "reward" in record, f"Missing reward in rollout: {record}"
                    assert "prompt" in record, f"Missing prompt in rollout: {record}"
                    assert "completion" in record, f"Missing completion in rollout: {record}"
                    assert "__metadata__" in record, f"Missing __metadata__ in rollout: {record}"
                    assert "step" in record["__metadata__"], f"Missing step in __metadata__: {record}"
                    return  # One record is enough

    def test_rollout_filter_applied(self, rollout_dir):
        """With filter_top_bottom(k=2), each step should have at most 4 records."""
        files = glob(os.path.join(rollout_dir, "*.jsonl"))
        records_per_step: dict[int, int] = {}
        for fp in files:
            with open(fp) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    step = record["__metadata__"]["step"]
                    records_per_step[step] = records_per_step.get(step, 0) + 1

        for step, count in records_per_step.items():
            assert count <= 4, f"Step {step} has {count} records (expected <=4 with k=2)"
