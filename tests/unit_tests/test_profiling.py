# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock

import torch

from torchtitan.tools.profiling import (
    CommsComputeOverlapAnalyzer,
    MemoryProfiler,
    ProfileAnalyzer,
    Profiler,
)


class TestProfileAnalyzerABC(unittest.TestCase):
    def test_profile_analyzer_is_abstract(self):
        """ProfileAnalyzer cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            ProfileAnalyzer()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_analyze(self):
        """A subclass that does not implement analyze() is also abstract."""
        class Incomplete(ProfileAnalyzer):
            pass

        with self.assertRaises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_concrete_subclass_is_instantiable(self):
        """A fully-implemented subclass can be instantiated."""
        class MyAnalyzer(ProfileAnalyzer):
            def analyze(self, prof: torch.profiler.profile) -> None:
                pass

        analyzer = MyAnalyzer()
        self.assertIsInstance(analyzer, ProfileAnalyzer)


class TestCommsComputeOverlapAnalyzer(unittest.TestCase):
    def _make_mock_prof(self, key_averages=None, events=None):
        prof = MagicMock(spec=torch.profiler.profile)
        prof.key_averages.return_value = key_averages or []
        prof.events.return_value = events or []
        return prof

    def test_analyze_no_nccl_kernels_logs_skip(self):
        """When no NCCL kernels are present, analyze() returns without error."""
        evt = MagicMock()
        evt.key = "aten::mm"
        evt.self_device_time_total = 1000.0
        prof = self._make_mock_prof(key_averages=[evt])

        analyzer = CommsComputeOverlapAnalyzer()
        # Should not raise; logs a "Skipping" message.
        analyzer.analyze(prof)

    def test_analyze_empty_key_averages(self):
        """Empty key_averages → no NCCL time → early return without error."""
        prof = self._make_mock_prof(key_averages=[])
        analyzer = CommsComputeOverlapAnalyzer()
        analyzer.analyze(prof)

    def test_analyze_nccl_and_compute_kernels(self):
        """With both NCCL and compute kernels present, analyze() runs to completion."""
        nccl_evt = MagicMock()
        nccl_evt.key = "nccl:all_reduce"
        nccl_evt.self_device_time_total = 5000.0

        compute_evt = MagicMock()
        compute_evt.key = "aten::mm"
        compute_evt.self_device_time_total = 8000.0

        prof = self._make_mock_prof(key_averages=[nccl_evt, compute_evt])
        analyzer = CommsComputeOverlapAnalyzer()
        # Should not raise; logs the overlap report.
        analyzer.analyze(prof)

    def test_get_trace_duration_empty_events(self):
        """_get_trace_duration_us() returns 0.0 for empty events."""
        prof = self._make_mock_prof(events=[])
        analyzer = CommsComputeOverlapAnalyzer()
        self.assertEqual(analyzer._get_trace_duration_us(prof), 0.0)

    def test_get_trace_duration_events_exception(self):
        """_get_trace_duration_us() returns 0.0 when events() raises."""
        prof = MagicMock(spec=torch.profiler.profile)
        prof.events.side_effect = RuntimeError("no trace")
        analyzer = CommsComputeOverlapAnalyzer()
        self.assertEqual(analyzer._get_trace_duration_us(prof), 0.0)


class TestProfilerConfig(unittest.TestCase):
    def test_default_field_values(self):
        cfg = Profiler.Config()
        self.assertFalse(cfg.enable_profiling)
        self.assertEqual(cfg.save_traces_folder, "profile_traces")
        self.assertEqual(cfg.profile_freq, 10)
        self.assertEqual(cfg.profiler_active, 1)
        self.assertEqual(cfg.profiler_warmup, 3)
        self.assertFalse(cfg.enable_memory_snapshot)
        self.assertEqual(cfg.save_memory_snapshot_folder, "memory_snapshot")
        self.assertFalse(cfg.enable_overlap_analysis)

    def test_custom_field_values(self):
        cfg = Profiler.Config(
            enable_profiling=True,
            save_traces_folder="my_traces",
            profile_freq=50,
        )
        self.assertTrue(cfg.enable_profiling)
        self.assertEqual(cfg.save_traces_folder, "my_traces")
        self.assertEqual(cfg.profile_freq, 50)

    def test_build_returns_profiler_instance(self):
        """Profiler.Config.build() auto-wires to Profiler via Configurable."""
        cfg = Profiler.Config()
        profiler = cfg.build()
        self.assertIsInstance(profiler, Profiler)


class TestProfilerInit(unittest.TestCase):
    def test_no_analyzers_by_default(self):
        """Profiler with default config has an empty analyzers list."""
        profiler = Profiler(Profiler.Config())
        self.assertEqual(profiler._analyzers, [])

    def test_enable_overlap_analysis_adds_comms_compute_overlap_analyzer(self):
        """enable_overlap_analysis=True adds CommsComputeOverlapAnalyzer automatically."""
        profiler = Profiler(Profiler.Config(enable_overlap_analysis=True))
        self.assertEqual(len(profiler._analyzers), 1)
        self.assertIsInstance(profiler._analyzers[0], CommsComputeOverlapAnalyzer)

    def test_default_runtime_attrs(self):
        """Profiler initializes runtime attrs to safe defaults."""
        profiler = Profiler(Profiler.Config())
        self.assertEqual(profiler._global_step, 0)
        self.assertEqual(profiler._base_folder, "")
        self.assertEqual(profiler._leaf_folder, "")
        self.assertIsNone(profiler.torch_profiler)
        self.assertIsNone(profiler.memory_profiler)


class TestProfilerDisabledPaths(unittest.TestCase):
    """Tests for the no-op / disabled paths that require no GPU."""

    def test_build_torch_profiler_disabled_returns_none(self):
        """build_torch_profiler returns None when profiling is disabled."""
        profiler = Profiler(Profiler.Config(enable_profiling=False))
        result = profiler.build_torch_profiler(
            global_step=0, base_folder="/tmp", leaf_folder=""
        )
        self.assertIsNone(result)

    def test_build_memory_profiler_disabled_returns_none(self):
        """build_memory_profiler returns None when memory snapshot is disabled."""
        profiler = Profiler(Profiler.Config(enable_memory_snapshot=False))
        result = profiler.build_memory_profiler(
            global_step=0, base_folder="/tmp", leaf_folder=""
        )
        self.assertIsNone(result)

    def test_active_returns_profiler_itself(self):
        """active() returns the Profiler instance (self)."""
        profiler = Profiler(Profiler.Config())
        result = profiler.active(global_step=5, base_folder="/tmp", leaf_folder="rank0")
        self.assertIs(result, profiler)

    def test_active_stores_runtime_args(self):
        """active() stores global_step, base_folder, leaf_folder on the instance."""
        profiler = Profiler(Profiler.Config())
        profiler.active(global_step=42, base_folder="/data", leaf_folder="sub")
        self.assertEqual(profiler._global_step, 42)
        self.assertEqual(profiler._base_folder, "/data")
        self.assertEqual(profiler._leaf_folder, "sub")

    def test_active_as_context_manager_step_is_noop(self):
        """With everything disabled, active() as context manager and step() don't raise."""
        profiler = Profiler(Profiler.Config())
        with profiler.active(global_step=0, base_folder="/tmp") as prof:
            self.assertIs(prof, profiler)
            self.assertIsNone(prof.torch_profiler)
            self.assertIsNone(prof.memory_profiler)
            prof.step()
            prof.step()

    def test_active_default_args(self):
        """active() works with all default keyword args."""
        profiler = Profiler(Profiler.Config())
        with profiler.active() as prof:
            prof.step()

    def test_step_noop_when_both_profilers_none(self):
        """step() is a no-op when torch_profiler and memory_profiler are both None."""
        profiler = Profiler(Profiler.Config())
        # should not raise even without entering the context manager
        profiler.step()
        profiler.step()

    def test_exit_resets_profiler_attrs(self):
        """After __exit__, torch_profiler and memory_profiler are reset to None."""
        profiler = Profiler(Profiler.Config())
        with profiler.active():
            pass
        self.assertIsNone(profiler.torch_profiler)
        self.assertIsNone(profiler.memory_profiler)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestProfilerEnabledPathsGPU(unittest.TestCase):
    """GPU-dependent tests — skipped automatically when no GPU is present."""

    def test_build_torch_profiler_returns_active_handle(self):
        import tempfile

        profiler = Profiler(
            Profiler.Config(
                enable_profiling=True,
                profile_freq=4,
                profiler_warmup=1,
                profiler_active=1,
            )
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with profiler.active(global_step=0, base_folder=tmpdir):
                self.assertIsNotNone(profiler.torch_profiler)


if __name__ == "__main__":
    unittest.main()
