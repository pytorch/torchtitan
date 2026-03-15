# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Post-training analysis: Chrome Trace (Gantt chart) from system JSONL.

Usage:
    python -m torchtitan.observability.analysis <system_logs_dir> <output_trace.json>

    View: open the output in chrome://tracing or https://ui.perfetto.dev
"""

import json
import os
import sys
from glob import glob


def load_all_records(log_dir: str) -> list[dict]:
    """Load all JSONL records from a system_logs directory."""
    records = []
    for path in sorted(glob(os.path.join(log_dir, "*.jsonl"))):
        source = os.path.basename(path).replace(".jsonl", "")
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    r["_source_file"] = source
                    records.append(r)
    return records


def generate_gantt_trace(log_dir: str, output_path: str) -> dict:
    """Convert system_logs/*.jsonl (from record_span) to Chrome Trace JSON.

    Reads system JSONL files only — NOT experiment JSONL. For experiment
    metrics, read experiment_logs/*.jsonl directly.

    Each source file becomes a process, each rank a thread. Start/end
    events become duration spans visible as a Gantt chart in Perfetto
    (https://ui.perfetto.dev) or chrome://tracing.

    Args:
        log_dir: Path to the system_logs/ directory.
        output_path: Where to write the Chrome Trace JSON file.

    Returns:
        The trace dict (also written to output_path).
    """
    records = load_all_records(log_dir)
    if not records:
        print(f"No records found in {log_dir}")
        return {"traceEvents": []}

    # Map source files to process IDs
    sources = sorted(set(r["_source_file"] for r in records))
    source_to_pid = {s: i for i, s in enumerate(sources)}

    events = []

    # Process name metadata
    for source, pid in source_to_pid.items():
        events.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": pid,
                "tid": 0,
                "args": {"name": source},
            }
        )

    # Collect start events to pair with their end events for "X" (complete) format.
    # Key: (name, pid, tid) → start timestamp
    pending_starts: dict[tuple, dict] = {}

    for r in records:
        normal = r.get("normal", {})
        event_type = normal.get("log_type_name", "")
        # Prefer microsecond timestamps; fall back to milliseconds for older logs
        int_fields = r.get("int", {})
        time_us = int_fields.get("time_us", int_fields.get("time_ms", 0) * 1000)
        pid = source_to_pid[r["_source_file"]]
        rank = r.get("int", {}).get("rank", 0)
        step = r.get("int", {}).get("step")

        if event_type.endswith("_start"):
            type_name = event_type.removesuffix("_start")
            display_name = type_name or normal.get("event_name", type_name)
            pending_starts[(type_name, pid, rank)] = {
                "ts": time_us, "step": step, "display_name": display_name,
            }
        elif event_type.endswith("_end"):
            type_name = event_type.removesuffix("_end")
            duration_ms = r.get("double", {}).get("value", 0)
            duration_us = duration_ms * 1000
            start = pending_starts.pop((type_name, pid, rank), None)
            start_ts = start["ts"] if start else time_us
            display_name = (start or {}).get("display_name", type_name)
            events.append(
                {
                    "name": display_name,
                    "ph": "X",
                    "ts": start_ts,
                    "dur": duration_us,
                    "pid": pid,
                    "tid": rank,
                    "args": {"step": step, "duration_ms": f"{duration_ms:.2f}"},
                }
            )
        elif event_type == "metric_value":
            event_name = normal.get("event_name", "metric")
            value = r.get("double", {}).get("value", 0)
            events.append(
                {
                    "name": f"{event_name}={value:.4f}",
                    "ph": "i",
                    "ts": time_us,
                    "pid": pid,
                    "tid": rank,
                    "s": "t",
                    "args": {"step": step},
                }
            )

    trace = {"traceEvents": events}
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(trace, f, indent=2)

    print(f"Chrome Trace: {output_path}")
    print(f"  {len(events)} events from {len(sources)} sources")
    print("  View in: chrome://tracing or https://ui.perfetto.dev")
    return trace


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    generate_gantt_trace(sys.argv[1], sys.argv[2])
