# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import importlib.util
import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from torchtitan.observability.always_on_profiler import (
    AlwaysOnProfiler,
    AlwaysOnProfilerConfig,
)


def _eager_spans(*starts: int) -> tuple[np.ndarray, ...]:
    start = np.array(starts, dtype=np.int64)
    return np.zeros(len(starts), np.uint64), start, start + 5, np.full(len(starts), 7)


@unittest.skipUnless(importlib.util.find_spec("cupti"), "requires cupti-python")
class TestAlwaysOnProfiler(unittest.TestCase):
    def test_buckets_spans_by_step_and_keeps_recent_steps(self):
        observer = mock.Mock(available=True)
        # Step boundaries in the record clock: the start, then the end of steps 1-3.
        observer.now_record_ns.side_effect = [0, 100, 200, 300]
        observer.drain.side_effect = [
            _eager_spans(10, 50),  # step 1
            _eager_spans(120, 250),  # step 2, and step 3 still running
            _eager_spans(260),  # step 3
        ]
        observer.convert_time_array.side_effect = lambda values: values
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch(
                "torch.profiler._cuspy.observers.node_timer.NodeTimerObserver",
                return_value=observer,
            ),
            mock.patch("torch.cuda.graphs.register_graph_instantiate_hook"),
            mock.patch("torch.cuda.current_device", return_value=0),
            mock.patch("torch.distributed.get_rank", return_value=0),
        ):
            profiler = AlwaysOnProfiler(
                AlwaysOnProfilerConfig(num_steps=2, dump_freq=3),
                trace_dir=tmpdir,
                global_step=0,
            )
            for _ in range(3):
                profiler.step()
            with gzip.open(
                os.path.join(tmpdir, "iteration_3", "rank0_trace.json.gz"), "rt"
            ) as f:
                events = json.load(f)["traceEvents"]

        # Step 1 fell out of the 2-step window, and the dump skips step 3, the
        # newest finished step, whose kernels may still be running.
        self.assertEqual([e["args"]["step"] for e in events], [2])
        self.assertEqual(events[0]["ts"], 0.12)  # microseconds
        self.assertEqual(sorted(profiler._spans), [2, 3])


if __name__ == "__main__":
    unittest.main()
