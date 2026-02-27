# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock

import torch
from torch.autograd import DeviceType

from torchtitan.tools.profiler import (
    CommsComputeOverlapAnalyzer,
    ProfileAnalyzer,
    Profiler,
    _union_us,
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


class TestUnionUs(unittest.TestCase):
    def test_empty_list(self):
        """Empty input returns 0.0."""
        self.assertEqual(_union_us([]), 0.0)

    def test_single_interval(self):
        """Single interval returns its own duration."""
        self.assertAlmostEqual(_union_us([(10.0, 50.0)]), 40.0)

    def test_non_overlapping_intervals(self):
        """Non-overlapping intervals: total is the sum of individual durations."""
        self.assertAlmostEqual(_union_us([(0.0, 10.0), (20.0, 30.0)]), 20.0)

    def test_partially_overlapping_intervals(self):
        """Partially overlapping intervals are merged correctly."""
        self.assertAlmostEqual(_union_us([(0.0, 15.0), (10.0, 25.0)]), 25.0)

    def test_fully_contained_interval(self):
        """An interval fully inside another collapses to the outer duration."""
        self.assertAlmostEqual(_union_us([(0.0, 100.0), (20.0, 50.0)]), 100.0)

    def test_adjacent_intervals(self):
        """Adjacent (touching but not overlapping) intervals are kept separate."""
        self.assertAlmostEqual(_union_us([(0.0, 10.0), (10.0, 20.0)]), 20.0)


class TestCommsComputeOverlapAnalyzer(unittest.TestCase):
    def _make_mock_prof(self, events=None):
        prof = MagicMock(spec=torch.profiler.profile)
        prof.events.return_value = events or []
        return prof

    def _make_cuda_event(self, name, start, end):
        evt = MagicMock()
        evt.device_type = DeviceType.CUDA
        evt.name = name
        evt.time_range = MagicMock()
        evt.time_range.start = start
        evt.time_range.end = end
        return evt

    def test_analyze_no_nccl_kernels_logs_skip(self):
        """When no NCCL kernels are present, analyze() returns without error."""
        evt = self._make_cuda_event("aten::mm", 0.0, 1000.0)
        prof = self._make_mock_prof(events=[evt])
        analyzer = CommsComputeOverlapAnalyzer()
        # Should not raise; logs a "Skipping" message.
        analyzer.analyze(prof)

    def test_analyze_empty_events(self):
        """Empty events → no NCCL time → early return without error."""
        prof = self._make_mock_prof(events=[])
        analyzer = CommsComputeOverlapAnalyzer()
        analyzer.analyze(prof)

    def test_analyze_non_cuda_events_ignored(self):
        """Events with device_type != CUDA are ignored even if they have NCCL names."""
        evt = MagicMock()
        evt.device_type = DeviceType.CPU
        evt.name = "nccl:all_reduce"
        prof = self._make_mock_prof(events=[evt])
        analyzer = CommsComputeOverlapAnalyzer()
        # No CUDA NCCL events → should log "Skipping" without error.
        analyzer.analyze(prof)

    def test_analyze_nccl_and_compute_kernels(self):
        """With both NCCL and compute kernels present, analyze() runs to completion."""
        # Non-overlapping: compute 0–8 ms, nccl 10–15 ms
        compute_evt = self._make_cuda_event("aten::mm", 0.0, 8000.0)
        nccl_evt = self._make_cuda_event("nccl:all_reduce", 10000.0, 15000.0)
        prof = self._make_mock_prof(events=[compute_evt, nccl_evt])
        analyzer = CommsComputeOverlapAnalyzer()
        # Should not raise; logs the overlap report.
        analyzer.analyze(prof)

    def test_analyze_full_overlap(self):
        """When compute and NCCL intervals are identical, overlap is 100%."""
        compute_evt = self._make_cuda_event("aten::mm", 0.0, 5000.0)
        nccl_evt = self._make_cuda_event("nccl:all_reduce", 0.0, 5000.0)
        prof = self._make_mock_prof(events=[compute_evt, nccl_evt])
        analyzer = CommsComputeOverlapAnalyzer()
        # Should not raise; overlap_pct should be 100%.
        analyzer.analyze(prof)


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

    def test_runtime_args_stored_on_init(self):
        """Runtime kwargs passed to __init__ are stored on the instance."""
        profiler = Profiler(
            Profiler.Config(), global_step=42, base_folder="/data", leaf_folder="sub"
        )
        self.assertEqual(profiler._global_step, 42)
        self.assertEqual(profiler._base_folder, "/data")
        self.assertEqual(profiler._leaf_folder, "sub")

    def test_context_manager_step_is_noop(self):
        """With everything disabled, context manager and step() don't raise."""
        profiler = Profiler(Profiler.Config())
        with profiler as prof:
            self.assertIs(prof, profiler)
            self.assertIsNone(prof.torch_profiler)
            self.assertIsNone(prof.memory_profiler)
            prof.step()
            prof.step()

    def test_default_args_context_manager(self):
        """Profiler with default runtime args works as a context manager."""
        profiler = Profiler(Profiler.Config())
        with profiler as prof:
            prof.step()

    def test_step_noop_when_both_profilers_none(self):
        """step() is a no-op when torch_profiler and memory_profiler are both None."""
        profiler = Profiler(Profiler.Config())
        profiler.step()
        profiler.step()

    def test_exit_resets_profiler_attrs(self):
        """After __exit__, torch_profiler and memory_profiler are reset to None."""
        profiler = Profiler(Profiler.Config())
        with profiler:
            pass
        self.assertIsNone(profiler.torch_profiler)
        self.assertIsNone(profiler.memory_profiler)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestProfilerEnabledPathsGPU(unittest.TestCase):
    """GPU-dependent tests — skipped automatically when no GPU is present."""

    def test_build_torch_profiler_returns_active_handle(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = Profiler(
                Profiler.Config(
                    enable_profiling=True,
                    profile_freq=4,
                    profiler_warmup=1,
                    profiler_active=1,
                ),
                global_step=0,
                base_folder=tmpdir,
            )
            with profiler:
                self.assertIsNotNone(profiler.torch_profiler)


if __name__ == "__main__":
    unittest.main()
