# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for rollout_logger.py."""

import json
import os

import pytest

from torchtitan.observability.rollout_logger import RolloutLogger


class TestRolloutLogger:
    def test_writes_jsonl(self, tmp_path):
        logger = RolloutLogger(output_dir=str(tmp_path))
        logger.log(
            [{"prompt": "hello", "reward": 0.5}, {"prompt": "world", "reward": 0.3}],
            metadata={"step": 1},
        )
        logger.close()

        with open(tmp_path / "rollouts.jsonl") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 2
        assert lines[0]["prompt"] == "hello"
        assert lines[0]["__metadata__"]["step"] == 1

    def test_metadata_nested_under_key(self, tmp_path):
        logger = RolloutLogger(output_dir=str(tmp_path))
        logger.log([{"x": 1}], metadata={"step": 42, "epoch": 3})
        logger.close()

        with open(tmp_path / "rollouts.jsonl") as f:
            record = json.loads(f.readline())
        assert record["__metadata__"]["step"] == 42
        assert record["__metadata__"]["epoch"] == 3
        assert record["x"] == 1
        # metadata doesn't pollute top-level keys
        assert "step" not in record
        assert "epoch" not in record

    def test_no_metadata(self, tmp_path):
        logger = RolloutLogger(output_dir=str(tmp_path))
        logger.log([{"x": 1}])
        logger.close()

        with open(tmp_path / "rollouts.jsonl") as f:
            record = json.loads(f.readline())
        assert record == {"x": 1}
        assert "__metadata__" not in record

    def test_empty_records_noop(self, tmp_path):
        logger = RolloutLogger(output_dir=str(tmp_path))
        logger.log([], metadata={"step": 1})
        logger.close()

        filepath = tmp_path / "rollouts.jsonl"
        assert filepath.stat().st_size == 0

    def test_filter_fn_applied(self, tmp_path):
        logger = RolloutLogger(output_dir=str(tmp_path))
        records = [{"id": i, "reward": i * 0.1} for i in range(10)]
        logger.log(records, metadata={"step": 1}, filter_fn=lambda r: r[:3])
        logger.close()

        with open(tmp_path / "rollouts.jsonl") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 3

    def test_custom_filename(self, tmp_path):
        logger = RolloutLogger(output_dir=str(tmp_path), filename="custom.jsonl")
        logger.log([{"x": 1}], metadata={"step": 1})
        logger.close()

        assert os.path.exists(tmp_path / "custom.jsonl")

    def test_append_across_calls(self, tmp_path):
        logger = RolloutLogger(output_dir=str(tmp_path))
        logger.log([{"step_data": "a"}], metadata={"step": 1})
        logger.log([{"step_data": "b"}], metadata={"step": 2})
        logger.close()

        with open(tmp_path / "rollouts.jsonl") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 2
