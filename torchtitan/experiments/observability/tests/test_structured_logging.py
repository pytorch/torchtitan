# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Integration tests for structured logging in toy_spmd.

Verifies system JSONL output from a toy_spmd run. These tests inspect
output files — they do not import internal modules.

Prerequisites: run toy_spmd first to generate output files.
"""

import json
import os
from glob import glob

import pytest

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs", "toy_spmd"
)


@pytest.fixture
def system_logs_dir():
    """Path to system_logs from a prior toy_spmd run."""
    path = os.path.join(OUTPUT_DIR, "system_logs")
    if not os.path.isdir(path):
        pytest.fail(
            f"No system_logs at {path}. Run toy_spmd first to generate outputs."
        )
    return path


class TestSystemJSONL:
    def test_system_jsonl_files_exist(self, system_logs_dir):
        files = glob(os.path.join(system_logs_dir, "*.jsonl"))
        assert len(files) >= 1, "Expected at least one system JSONL file"

    def test_system_jsonl_has_valid_structure(self, system_logs_dir):
        files = glob(os.path.join(system_logs_dir, "*.jsonl"))
        for fp in files:
            with open(fp) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    assert "int" in record
                    assert "normal" in record
                    assert "double" in record
                    assert "normvector" in record
                    assert "rank" in record["int"]

    def test_system_jsonl_has_step_events(self, system_logs_dir):
        files = glob(os.path.join(system_logs_dir, "*.jsonl"))
        event_types = set()
        for fp in files:
            with open(fp) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    et = record.get("normal", {}).get("log_type_name")
                    if et:
                        event_types.add(et)
        assert "step_start" in event_types, f"Missing step_start in {event_types}"
        assert "step_end" in event_types, f"Missing step_end in {event_types}"

    def test_system_jsonl_has_caller_field(self, system_logs_dir):
        files = glob(os.path.join(system_logs_dir, "*.jsonl"))
        for fp in files:
            with open(fp) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    caller = record.get("normal", {}).get("caller")
                    assert caller is not None, "Missing caller field"
                    assert ":" in caller, f"caller field missing colon: {caller}"
                    return  # One check is enough


class TestChromeTrace:
    def test_trace_json_exists(self):
        trace_path = os.path.join(OUTPUT_DIR, "analysis", "system_metrics_gantt.json")
        if not os.path.exists(trace_path):
            pytest.fail("No analysis/system_metrics_gantt.json. Run toy_spmd first.")
        with open(trace_path) as f:
            trace = json.load(f)
        assert "traceEvents" in trace
        assert len(trace["traceEvents"]) > 0
