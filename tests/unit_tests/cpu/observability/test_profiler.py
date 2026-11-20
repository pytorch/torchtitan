# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from torchtitan.observability.profiler import (
    CuspyTraceProfiler,
    KinetoTraceProfiler,
    MemorySnapshotProfiler,
    Profiler,
)


class TestProfilerDisabled(unittest.TestCase):
    def test_context_manager_step_is_noop(self):
        """With nothing enabled, the context manager and step() need no process group."""
        profiler = Profiler(Profiler.Config())
        with profiler as prof:
            self.assertIs(prof, profiler)
            self.assertIsNone(prof.trace)
            self.assertIsNone(prof.memory_snapshot)
            prof.step()
            prof.step()


class TestProfilerEnabled(unittest.TestCase):
    """Tests for enabled profilers -- uses mocked distributed rank."""

    def setUp(self):
        self.patcher_rank = mock.patch("torch.distributed.get_rank", return_value=0)
        self.patcher_rank.start()

    def tearDown(self):
        self.patcher_rank.stop()

    def test_kineto_backend_built_and_closed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = Profiler(
                Profiler.Config(
                    trace=KinetoTraceProfiler.Config(freq=4, warmup=1, active=1)
                ),
                global_step=0,
                base_folder=tmpdir,
            )
            with profiler:
                self.assertIsInstance(profiler.trace, KinetoTraceProfiler)
                profiler.step()
            self.assertIsNone(profiler.trace)

    def test_neuron_adds_privateuse1_activity_and_stacks(self):
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch("torch.xpu.is_available", return_value=False),
            mock.patch("torchtitan.observability.profiler.device_type", "neuron"),
            mock.patch("torch.profiler.profile") as profile,
        ):
            KinetoTraceProfiler.Config(freq=4, warmup=1, active=1).build(
                global_step=0, base_folder=tmpdir, leaf_folder="", rank=0
            )
        self.assertEqual(
            profile.call_args.kwargs["activities"],
            [
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.PrivateUse1,
            ],
        )
        self.assertTrue(profile.call_args.kwargs["with_stack"])

    def test_memory_snapshot_dumps_every_freq_steps_and_on_oom(self):
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch("torchtitan.observability.profiler.device_module") as device,
            mock.patch("torchtitan.observability.profiler.pickle.dump"),
        ):
            device.memory._snapshot.return_value = {}
            profiler = Profiler(
                Profiler.Config(memory_snapshot=MemorySnapshotProfiler.Config(freq=3)),
                global_step=1,
                base_folder=tmpdir,
            )
            with self.assertRaises(RuntimeError):
                with profiler:
                    for _ in range(3):
                        profiler.step()
                    try:
                        raise torch.OutOfMemoryError("oom")
                    except torch.OutOfMemoryError as oom:
                        # Pipeline parallelism wraps the OOM in a RuntimeError.
                        raise RuntimeError("stage failed") from oom
            self.assertEqual(
                sorted(os.listdir(os.path.join(tmpdir, "profiling/memory_snapshot"))),
                ["step_000000000003", "step_000000000004_exit"],
            )

    def test_cuspy_raises_when_environment_unsupported(self):
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch(
                "torchtitan.observability.profiler.cuspy_unavailable_reasons",
                return_value=["reason one", "reason two"],
            ),
        ):
            with self.assertRaisesRegex(
                ValueError,
                re.escape("Cuspy cannot run here:\n  - reason one\n  - reason two"),
            ):
                CuspyTraceProfiler.Config(freq=4, warmup=1, active=1).build(
                    global_step=0, base_folder=tmpdir, leaf_folder="", rank=0
                )

    def test_cuspy_trace_file(self):
        for trace_format, file_name in (
            ("json", "rank0_trace.json.gz"),
            ("pftrace", "rank0_trace.pftrace.gz"),
        ):
            with (
                self.subTest(trace_format=trace_format),
                tempfile.TemporaryDirectory() as tmpdir,
                mock.patch.object(
                    CuspyTraceProfiler,
                    "profile_activities",
                    return_value=["cuspy-activity"],
                ),
                mock.patch("torch.profiler.profile") as profile,
            ):
                CuspyTraceProfiler.Config(
                    freq=4, warmup=1, active=1, trace_format=trace_format
                ).build(global_step=0, base_folder=tmpdir, leaf_folder="", rank=0)
                kwargs = profile.call_args.kwargs
                self.assertEqual(kwargs["activities"], ["cuspy-activity"])
                prof = mock.Mock(step_num=4)
                kwargs["on_trace_ready"](prof)
                prof.export_chrome_trace.assert_called_once_with(
                    os.path.join(tmpdir, "profiling/traces", "iteration_4", file_name)
                )

    @unittest.skipUnless(
        hasattr(torch.profiler, "CuspyConfig"), "torch build has no Cuspy"
    )
    def test_cuspy_activity_config(self):
        """Builds real torch Cuspy configs, so torch API changes fail here."""
        config = CuspyTraceProfiler.Config(
            enable_event_node_ids=True,
            performance_metrics=["sm__cycles_active.avg"],
            pm_sampling_interval_ms=0.5,
        )
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch(
                "torchtitan.observability.profiler.cuspy_unavailable_reasons",
                return_value=[],
            ),
            mock.patch("torch.cuda.graph_annotations.is_available", return_value=True),
        ):
            _, activity = CuspyTraceProfiler.profile_activities(
                SimpleNamespace(_config=config)
            )
        cuspy_config, pm_config = activity[
            torch.profiler.ProfilerActivity.CUDA
        ].profiler_configs
        self.assertTrue(cuspy_config.enable_event_node_ids)
        self.assertEqual(pm_config.metric_names, ["sm__cycles_active.avg"])
        self.assertEqual(pm_config.sampling_interval_ms, 0.5)


if __name__ == "__main__":
    unittest.main()
