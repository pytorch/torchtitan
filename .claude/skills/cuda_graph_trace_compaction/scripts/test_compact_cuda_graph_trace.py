#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the CUDA graph trace compactor.

The compactor lives beside this file rather than inside the torchtitan
package, so run it directly:

    pytest .claude/skills/cuda_graph_trace_compaction/scripts/
"""

import sys
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from compact_cuda_graph_trace import compact_trace, merge_pp_traces


class TestCompactCudaGraphTrace(unittest.TestCase):
    def test_merges_aligned_pp_ranks_and_connects_send_recv(self) -> None:
        send_trace = self._rank_trace(
            rank=0,
            base_time_ns=1_000_000,
            events=[
                self._thread_metadata(pid=100, tid=100, name="thread 100 (python) 100"),
                self._kernel(
                    "ncclDevKernel_AllGather",
                    stream=6,
                    timestamp=1,
                    duration=4,
                    annotation="PP:0UNSHARD",
                ),
                self._kernel(
                    "ncclDevKernel_AllGather",
                    stream=7,
                    timestamp=2,
                    duration=2,
                    annotation="PP:0UNSHARD",
                ),
                self._kernel(
                    "compute",
                    stream=2,
                    timestamp=4,
                    duration=1,
                    annotation="PP:0F0",
                ),
                self._kernel(
                    "compute",
                    stream=3,
                    timestamp=4.5,
                    duration=4,
                    annotation="PP:0F0",
                ),
                self._kernel("compute", stream=5, timestamp=6),
                self._kernel(
                    "compute",
                    stream=2,
                    timestamp=8,
                    annotation="PP:0F0",
                ),
                self._memcpy(
                    stream=4,
                    timestamp=3.5,
                    duration=2,
                    annotation="PP:0F0",
                ),
                self._kernel(
                    "ncclDevKernel_SendRecv",
                    stream=1,
                    timestamp=10,
                    duration=4,
                    annotation="PP:0SEND_F0",
                ),
                self._flow(stream=1, timestamp=10),
            ],
        )
        recv_trace = self._rank_trace(
            rank=4,
            base_time_ns=1_001_000,
            events=[
                self._kernel(
                    "ncclDevKernel_AllGather",
                    stream=6,
                    timestamp=1,
                    duration=4,
                    annotation="PP:1UNSHARD",
                ),
                self._kernel(
                    "ncclDevKernel_AllGather",
                    stream=7,
                    timestamp=2,
                    duration=2,
                    annotation="PP:1UNSHARD",
                ),
                self._kernel(
                    "ncclDevKernel_SendRecv",
                    stream=1,
                    timestamp=8,
                    duration=5,
                    annotation="PP:1RECV_F0",
                ),
                self._flow(stream=1, timestamp=8),
                self._kernel(
                    "compute",
                    stream=2,
                    timestamp=14,
                    duration=1,
                    annotation="PP:1F0",
                ),
                self._kernel(
                    "compute",
                    stream=3,
                    timestamp=14.5,
                    duration=4,
                    annotation="PP:1F0",
                ),
                self._kernel("compute", stream=5, timestamp=16),
                self._kernel(
                    "compute",
                    stream=2,
                    timestamp=18,
                    annotation="PP:1F0",
                ),
                self._memcpy(
                    stream=4,
                    timestamp=13.5,
                    duration=2,
                    annotation="PP:1F0",
                ),
            ],
        )

        merged, summary = merge_pp_traces([recv_trace, send_trace])

        kernels = {
            event["args"]["name"]: event
            for event in merged["traceEvents"]
            if event.get("cat") == "kernel" and "name" in event.get("args", {})
        }
        send_recv_flows = [
            event
            for event in merged["traceEvents"]
            if event.get("cat") == "pp_send_recv"
        ]
        dependency_flows = [
            event
            for event in merged["traceEvents"]
            if event.get("cat") == "pp_compute_dependency"
        ]
        unshard_flows = [
            event
            for event in merged["traceEvents"]
            if event.get("cat") == "pp_unshard_dependency"
        ]
        annotations = {
            event["name"]: event
            for event in merged["traceEvents"]
            if event.get("cat") == "gpu_user_annotation"
            and event.get("name") in {"PP:0F0", "PP:0SEND_F0", "PP:1RECV_F0", "PP:1F0"}
        }
        original_flows = [
            event for event in merged["traceEvents"] if event.get("cat") == "ac2g"
        ]
        thread_names = {
            event["args"]["name"]
            for event in merged["traceEvents"]
            if event.get("name") == "thread_name"
        }
        thread_names_by_lane = {
            (event["pid"], event["tid"]): event["args"]["name"]
            for event in merged["traceEvents"]
            if event.get("name") == "thread_name"
        }
        process_sort_indices = {
            event["pid"]: event["args"]["sort_index"]
            for event in merged["traceEvents"]
            if event.get("name") == "process_sort_index"
        }
        cpu_thread = next(
            event
            for event in merged["traceEvents"]
            if event.get("name") == "thread_name"
            and event.get("args", {}).get("name")
            == "PP rank 0 | thread 100 (python) 100"
        )
        self.assertEqual(10, kernels["PP:0SEND_F0"]["ts"])
        self.assertEqual(9.0, kernels["PP:1RECV_F0"]["ts"])
        self.assertNotEqual(
            kernels["PP:0SEND_F0"]["pid"], kernels["PP:1RECV_F0"]["pid"]
        )
        self.assertEqual(["s", "f"], [event["ph"] for event in send_recv_flows])
        self.assertLess(send_recv_flows[0]["ts"], send_recv_flows[1]["ts"])
        self.assertIsInstance(send_recv_flows[0]["id"], int)
        self.assertEqual(send_recv_flows[0]["id"], send_recv_flows[1]["id"])
        self.assertEqual(2, len({event["id"] for event in original_flows}))
        self.assertTrue(all(isinstance(event["id"], int) for event in original_flows))
        self.assertEqual(annotations["PP:0SEND_F0"]["tid"], send_recv_flows[0]["tid"])
        self.assertEqual(annotations["PP:1RECV_F0"]["tid"], send_recv_flows[1]["tid"])
        self.assertNotEqual(kernels["PP:0SEND_F0"]["tid"], send_recv_flows[0]["tid"])
        self.assertEqual(
            {"0F0 -> 0SEND_F0", "1RECV_F0 -> 1F0"},
            {event["name"] for event in dependency_flows},
        )
        compute_flow_endpoints = [
            event
            for event in dependency_flows
            if (event["ph"] == "s" and event["args"]["source_operation"] in {"F", "B"})
            or (
                event["ph"] == "f"
                and event["args"]["destination_operation"] in {"F", "B"}
            )
        ]
        self.assertEqual(2, len(compute_flow_endpoints))
        self.assertTrue(
            all(
                thread_names_by_lane[(event["pid"], event["tid"])].endswith(
                    "Compute 1 annotations"
                )
                for event in compute_flow_endpoints
            )
        )
        for event in compute_flow_endpoints:
            args = event["args"]
            endpoint = "source" if event["ph"] == "s" else "destination"
            operation = args[f"{endpoint}_operation"]
            stage = args[f"{endpoint}_stage"]
            annotation_name = f"PP:{stage}{operation}{args['microbatch']}"
            self.assertTrue(
                any(
                    annotation.get("pid") == event["pid"]
                    and annotation.get("tid") == event["tid"]
                    and annotation.get("name") == annotation_name
                    and annotation["ts"]
                    <= event["ts"]
                    <= annotation["ts"] + annotation["dur"]
                    for annotation in merged["traceEvents"]
                    if annotation.get("cat") == "gpu_user_annotation"
                )
            )
        self.assertLess(
            process_sort_indices[cpu_thread["pid"]],
            process_sort_indices[kernels["PP:0F0"]["pid"]],
        )
        self.assertIn("PP rank 0 | NCCL other", thread_names)
        self.assertIn("PP rank 1 | NCCL other", thread_names)
        self.assertEqual("baseTimeNanoseconds", summary["time_alignment"])
        self.assertEqual({0: 0.0, 4: 1.0}, summary["time_offsets_us"])
        self.assertEqual(1, summary["pp_send_recv_pairs"])
        self.assertEqual(0, summary["pp_send_recv_unmatched"])
        self.assertEqual(2, summary["pp_compute_dependency_pairs"])
        self.assertEqual(0, summary["pp_compute_dependency_unmatched"])
        self.assertEqual(
            {"0UNSHARD -> 0F0", "1UNSHARD -> 1F0"},
            {event["name"] for event in unshard_flows},
        )
        self.assertTrue(
            {
                thread_names_by_lane[(event["pid"], event["tid"])].rsplit(
                    " | ", maxsplit=1
                )[-1]
                for event in unshard_flows
                if event["ph"] == "s"
            }
            == {
                "NCCL all-gather 1 annotations",
                "NCCL all-gather 2 annotations",
            }
        )
        self.assertTrue(
            all(
                thread_names_by_lane[(event["pid"], event["tid"])].endswith(
                    "Compute 1 annotations"
                )
                for event in unshard_flows
                if event["ph"] == "f"
            )
        )
        for event in unshard_flows:
            args = event["args"]
            endpoint = "source" if event["ph"] == "s" else "destination"
            operation = args[f"{endpoint}_operation"]
            stage = args[f"{endpoint}_stage"]
            annotation_name = (
                f"PP:{stage}UNSHARD"
                if operation == "UNSHARD"
                else f"PP:{stage}{operation}{args['microbatch']}"
            )
            self.assertTrue(
                any(
                    annotation.get("pid") == event["pid"]
                    and annotation.get("tid") == event["tid"]
                    and annotation.get("name") == annotation_name
                    and annotation["ts"]
                    <= event["ts"]
                    <= annotation["ts"] + annotation["dur"]
                    for annotation in merged["traceEvents"]
                    if annotation.get("cat") == "gpu_user_annotation"
                )
            )
        for name in {"0UNSHARD -> 0F0", "1UNSHARD -> 1F0"}:
            destinations = [
                event
                for event in unshard_flows
                if event["name"] == name and event["ph"] == "f"
            ]
            self.assertEqual(2, len(destinations))
            self.assertEqual(
                1,
                len(
                    {
                        (event["pid"], event["tid"], event["ts"])
                        for event in destinations
                    }
                ),
            )
        self.assertEqual(4, summary["pp_unshard_compute_pairs"])
        self.assertEqual(0, summary["pp_unshard_compute_unmatched"])

    def test_compacts_single_rank_with_local_pp_flows(self) -> None:
        trace = self._rank_trace(
            rank=4,
            base_time_ns=1_000_000,
            events=[
                self._kernel(
                    "ncclDevKernel_AllGather",
                    stream=6,
                    timestamp=1,
                    duration=4,
                    annotation="PP:1UNSHARD",
                ),
                self._kernel(
                    "ncclDevKernel_AllGather",
                    stream=7,
                    timestamp=2,
                    duration=2,
                    annotation="PP:1UNSHARD",
                ),
                self._kernel(
                    "ncclDevKernel_SendRecv",
                    stream=1,
                    timestamp=4,
                    duration=3,
                    annotation="PP:1RECV_F0",
                ),
                self._kernel(
                    "compute",
                    stream=2,
                    timestamp=8,
                    duration=2,
                    annotation="PP:1F0",
                ),
                self._kernel(
                    "ncclDevKernel_SendRecv",
                    stream=1,
                    timestamp=11,
                    annotation="PP:1SEND_F0",
                ),
            ],
        )
        trace["distributedInfo"]["pg_config"] = [
            {
                "pg_name": "recorded-group-id",
                "pg_desc": "mesh_pp",
                "ranks": [0, 4],
            }
        ]

        compacted, summary = compact_trace(trace)

        events = compacted["traceEvents"]
        dependency_flows = [
            event for event in events if event.get("cat") == "pp_compute_dependency"
        ]
        unshard_flows = [
            event for event in events if event.get("cat") == "pp_unshard_dependency"
        ]
        thread_names = {
            (event["pid"], event["tid"]): event["args"]["name"]
            for event in events
            if event.get("name") == "thread_name"
        }
        self.assertFalse(any(event.get("cat") == "pp_send_recv" for event in events))
        self.assertEqual(
            {"1RECV_F0 -> 1F0", "1F0 -> 1SEND_F0"},
            {event["name"] for event in dependency_flows},
        )
        self.assertEqual(
            {
                "NCCL all-gather 1 annotations",
                "NCCL all-gather 2 annotations",
            },
            {
                thread_names[(event["pid"], event["tid"])]
                for event in unshard_flows
                if event["ph"] == "s"
            },
        )
        self.assertTrue(
            all(
                thread_names[(event["pid"], event["tid"])] == "Compute annotations"
                for event in unshard_flows
                if event["ph"] == "f"
            )
        )
        self.assertTrue(
            all(
                event["args"]["source_pp_rank"] == 1
                and event["args"]["destination_pp_rank"] == 1
                for event in dependency_flows + unshard_flows
            )
        )
        self.assertEqual(1, summary["pp_rank"])
        self.assertEqual(2, summary["pp_compute_dependency_pairs"])
        self.assertEqual(0, summary["pp_compute_dependency_unmatched"])
        self.assertEqual(2, summary["pp_unshard_compute_pairs"])
        self.assertEqual(0, summary["pp_unshard_compute_unmatched"])

    def test_converts_neighbor_kernel_annotations_to_spans(self) -> None:
        trace = {
            "traceEvents": [
                self._kernel("compute", stream=1, timestamp=10, annotation="PP:F0"),
                self._flow(stream=1, timestamp=10),
                self._kernel(
                    "compute",
                    stream=1,
                    timestamp=12,
                    duration=2,
                    annotation="PP:F0",
                ),
                self._kernel("compute", stream=1, timestamp=15, annotation="PP:B0"),
                self._kernel("compute", stream=1, timestamp=17),
                self._kernel("compute", stream=1, timestamp=19, annotation="PP:B0"),
            ]
        }

        compacted, summary = compact_trace(trace)

        kernels = [
            event for event in compacted["traceEvents"] if event.get("cat") == "kernel"
        ]
        annotations = sorted(
            (
                event["name"],
                event["ts"],
                event["dur"],
            )
            for event in compacted["traceEvents"]
            if event.get("cat") == "gpu_user_annotation"
        )
        self.assertEqual(
            ["PP:F0", "PP:F0", "PP:B0", None, "PP:B0"],
            [event["args"].get("name") for event in kernels],
        )
        flow = next(
            event for event in compacted["traceEvents"] if event.get("cat") == "ac2g"
        )
        annotation_tids = {
            event["tid"]
            for event in compacted["traceEvents"]
            if event.get("cat") == "gpu_user_annotation"
        }
        self.assertEqual(kernels[0]["tid"], flow["tid"])
        self.assertNotIn(flow["tid"], annotation_tids)

        sort_indices = {
            event["tid"]: event["args"]["sort_index"]
            for event in compacted["traceEvents"]
            if event.get("name") == "thread_sort_index"
        }
        self.assertLess(
            sort_indices[next(iter(annotation_tids))],
            sort_indices[kernels[0]["tid"]],
        )
        self.assertEqual(
            [
                ("PP:B0", 15.0, 1.0),
                ("PP:B0", 19.0, 1.0),
                ("PP:F0", 10.0, 4.0),
            ],
            annotations,
        )
        self.assertEqual(4, summary["kernel_annotations_converted"])
        self.assertEqual(3, summary["kernel_annotation_blocks"])
        self.assertEqual(3, summary["gpu_annotations_retained"])

    def test_coalesces_replay_annotations_per_collective_lane(self) -> None:
        trace = {
            "traceEvents": [
                self._kernel("ncclDevKernel_AllGather", stream=1, timestamp=10),
                self._annotation(stream=1, timestamp=9, duration=5),
                self._kernel("ncclDevKernel_AllGather", stream=2, timestamp=12),
                self._annotation(stream=2, timestamp=11, duration=5),
                self._kernel("ncclDevKernel_ReduceScatter", stream=3, timestamp=10),
                self._annotation(stream=3, timestamp=9, duration=5),
                self._kernel("ncclDevKernel_ReduceScatter", stream=4, timestamp=12),
                self._annotation(stream=4, timestamp=11, duration=5),
            ]
        }

        compacted, summary = compact_trace(trace)

        labels = {
            event["tid"]: event["args"]["name"]
            for event in compacted["traceEvents"]
            if event.get("ph") == "M" and event.get("name") == "thread_name"
        }
        annotations_by_lane = {
            labels[event["tid"]]: event
            for event in compacted["traceEvents"]
            if event.get("cat") == "gpu_user_annotation"
        }
        self.assertEqual(
            {
                "NCCL all-gather annotations",
                "NCCL reduce-scatter annotations",
            },
            set(annotations_by_lane),
        )
        self.assertEqual(9.0, annotations_by_lane["NCCL all-gather annotations"]["ts"])
        self.assertEqual(7.0, annotations_by_lane["NCCL all-gather annotations"]["dur"])
        self.assertEqual(
            9.0, annotations_by_lane["NCCL reduce-scatter annotations"]["ts"]
        )
        self.assertEqual(
            7.0, annotations_by_lane["NCCL reduce-scatter annotations"]["dur"]
        )
        self.assertEqual(4, summary["gpu_annotations_mapped"])
        self.assertEqual(2, summary["gpu_annotations_coalesced"])
        self.assertEqual(2, summary["gpu_annotations_retained"])

    def test_preserves_distinct_and_disjoint_annotations(self) -> None:
        trace = {
            "traceEvents": [
                self._kernel("compute", stream=1, timestamp=10),
                self._annotation(stream=1, timestamp=9, duration=5),
                self._kernel("compute", stream=2, timestamp=12),
                self._annotation(stream=2, timestamp=11, duration=5),
                self._kernel("compute", stream=3, timestamp=14),
                self._annotation(
                    stream=3,
                    timestamp=13,
                    duration=3,
                    external_id=2,
                ),
                self._kernel("compute", stream=4, timestamp=30),
                self._annotation(stream=4, timestamp=30, duration=1),
            ]
        }

        compacted, summary = compact_trace(trace)

        annotations = [
            event
            for event in compacted["traceEvents"]
            if event.get("cat") == "gpu_user_annotation"
        ]
        annotation_intervals = sorted(
            (
                event["args"]["External id"],
                event["ts"],
                event["dur"],
            )
            for event in annotations
        )
        self.assertEqual(
            [(1, 9.0, 7.0), (1, 30, 1), (2, 13, 3)],
            annotation_intervals,
        )
        self.assertEqual(1, summary["gpu_annotations_coalesced"])
        self.assertEqual(3, summary["gpu_annotations_retained"])

    def _kernel(
        self,
        name: str,
        stream: int,
        timestamp: float,
        duration: float = 1,
        annotation: str | None = None,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {"stream": stream}
        if annotation is not None:
            args["name"] = annotation
        return {
            "name": name,
            "cat": "kernel",
            "ph": "X",
            "pid": 0,
            "tid": stream,
            "ts": timestamp,
            "dur": duration,
            "args": args,
        }

    def _annotation(
        self,
        stream: int,
        timestamp: float,
        duration: float,
        external_id: int = 1,
    ) -> dict[str, Any]:
        return {
            "name": "ProfilerStep#19",
            "cat": "gpu_user_annotation",
            "ph": "X",
            "pid": 0,
            "tid": stream,
            "ts": timestamp,
            "dur": duration,
            "args": {"External id": external_id},
        }

    def _memcpy(
        self,
        stream: int,
        timestamp: float,
        duration: float,
        annotation: str,
    ) -> dict[str, Any]:
        return {
            "name": "Memcpy DtoD",
            "cat": "gpu_memcpy",
            "ph": "X",
            "pid": 0,
            "tid": stream,
            "ts": timestamp,
            "dur": duration,
            "args": {"stream": stream, "name": annotation},
        }

    def _flow(self, stream: int, timestamp: float) -> dict[str, Any]:
        return {
            "name": "ac2g",
            "cat": "ac2g",
            "ph": "f",
            "pid": 0,
            "tid": stream,
            "ts": timestamp,
            "id": 1,
        }

    def _thread_metadata(self, pid: int, tid: int, name: str) -> dict[str, Any]:
        return {
            "name": "thread_name",
            "ph": "M",
            "pid": pid,
            "tid": tid,
            "args": {"name": name},
        }

    def _rank_trace(
        self,
        rank: int,
        base_time_ns: int,
        events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "baseTimeNanoseconds": base_time_ns,
            "displayTimeUnit": "ms",
            "distributedInfo": {
                "backend": "fake",
                "rank": rank,
                "pg_config": [
                    {
                        "pg_name": "torchtitan_real_pp",
                        "ranks": [0, 4],
                    }
                ],
            },
            "traceEvents": events,
        }


if __name__ == "__main__":
    unittest.main()
