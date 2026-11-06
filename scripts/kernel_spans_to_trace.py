# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Convert recorded kernel span steps into a Chrome trace for Perfetto.

Reads step files written by ``Profiler.Config.kernel_spans`` (see
``torchtitan/observability/kernel_spans.py``), one rank and many steps per
file, and joins each kernel span to its CUDA graph node, so slices carry the
kernel name and the node's graph annotations. Ranks become processes and CUDA
streams become threads; each step also gets a slice on a ``steps`` track
spanning its kernels.

Example::

    python scripts/kernel_spans_to_trace.py \\
        outputs/profiling/kernel_spans/rank{0,1}/steps_000000000101_000000000200.npz \\
        --steps 150-152 --output trace.json.gz

Open the output in https://ui.perfetto.dev.
"""

import argparse
import gzip
import json

import numpy as np

# Turns mangled C++ kernel names (e.g. _ZN2at6native...) into readable ones.
# TODO: private torch API; switch to a public demangler if torch exposes one.
from torch._C import _demangle

# Thread id for the per-rank track of step slices; CUDA stream ids are small.
_STEPS_TID = -1


def _parse_steps(spec: str) -> list[int]:
    """Parse ``"3,5-7"`` into ``[3, 5, 6, 7]``."""
    values: list[int] = []
    for part in spec.split(","):
        first, _, last = part.partition("-")
        values.extend(range(int(first), int(last or first) + 1))
    return values


def _file_events(path: str, steps: set[int]) -> tuple[list, set[int]]:
    """Return the trace events for ``steps`` in one rank's step file, and the
    steps found."""
    data = np.load(path)
    rank = int(data["rank"])
    node_table = json.loads(data["node_table"].tobytes())
    offsets = data["step_offsets"]
    events: list[dict] = [
        {
            "ph": "M",
            "name": "process_name",
            "pid": rank,
            "args": {"name": f"rank {rank}"},
        },
        {
            "ph": "M",
            "name": "thread_name",
            "pid": rank,
            "tid": _STEPS_TID,
            "args": {"name": "steps"},
        },
    ]
    found: set[int] = set()
    streams: set[int] = set()
    num_unnamed = 0
    for i, step in enumerate(data["steps"].tolist()):
        if step not in steps:
            continue
        found.add(step)
        rows = slice(offsets[i], offsets[i + 1])
        node_ids = data["graph_node_id"][rows].tolist()
        starts = data["start_ns"][rows].tolist()
        ends = data["end_ns"][rows].tolist()
        stream_ids = data["stream_id"][rows].tolist()
        for node_id, start, end, stream in zip(node_ids, starts, ends, stream_ids):
            streams.add(stream)
            node = node_table.get(str(node_id)) if node_id else None
            args: dict = {"graph_node_id": node_id}
            if node is None:
                name = "eager kernel" if node_id == 0 else "unknown graph node"
                num_unnamed += node_id != 0
            else:
                name = _demangle(node["kernel_name"] or node["node_type"])
                if node["grid"] is not None:
                    args["grid"], args["block"] = node["grid"], node["block"]
                for annotation in node["annotations"] or []:
                    if isinstance(annotation, dict):
                        args.update(annotation)
            events.append(
                {
                    "ph": "X",
                    "cat": "kernel",
                    "name": name,
                    "pid": rank,
                    "tid": stream,
                    "ts": start / 1e3,
                    "dur": (end - start) / 1e3,
                    "args": args,
                }
            )
        if starts:
            events.append(
                {
                    "ph": "X",
                    "cat": "step",
                    "name": f"step {step}",
                    "pid": rank,
                    "tid": _STEPS_TID,
                    "ts": min(starts) / 1e3,
                    "dur": (max(ends) - min(starts)) / 1e3,
                    "args": {"num_kernels": len(starts)},
                }
            )
    events.extend(
        {
            "ph": "M",
            "name": "thread_name",
            "pid": rank,
            "tid": stream,
            "args": {"name": f"stream {stream}"},
        }
        for stream in sorted(streams)
    )
    if num_unnamed:
        print(
            f"rank {rank}: {num_unnamed} graph kernel spans have no node table entry "
            "(their graph was instantiated before the recorder started)"
        )
    return events, found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "files", nargs="+", help="step files to convert, one per rank to show"
    )
    parser.add_argument(
        "--steps", required=True, help='steps to convert, e.g. "150-152,160"'
    )
    parser.add_argument("--output", required=True, help="output .json or .json.gz path")
    args = parser.parse_args()

    steps = set(_parse_steps(args.steps))
    events: list[dict] = []
    for path in args.files:
        file_events, found = _file_events(path, steps)
        if missing := sorted(steps - found):
            print(f"{path}: steps {missing} are not in this file")
        events.extend(file_events)

    opener = gzip.open if args.output.endswith(".gz") else open
    with opener(args.output, "wt") as f:
        json.dump({"traceEvents": events, "displayTimeUnit": "ms"}, f)
    print(f"wrote {sum(e['ph'] == 'X' for e in events)} slices to {args.output}")


if __name__ == "__main__":
    main()
