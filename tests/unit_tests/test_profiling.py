# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock

import torch

from torchtitan.tools.profiling import (
    OverlapAnalyzer,
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


class TestOverlapAnalyzer(unittest.TestCase):
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

        analyzer = OverlapAnalyzer()
        # Should not raise; logs a "Skipping" message.
        analyzer.analyze(prof)

    def test_analyze_empty_key_averages(self):
        """Empty key_averages → no NCCL time → early return without error."""
        prof = self._make_mock_prof(key_averages=[])
        analyzer = OverlapAnalyzer()
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
        analyzer = OverlapAnalyzer()
        # Should not raise; logs the overlap report.
        analyzer.analyze(prof)

    def test_get_trace_duration_empty_events(self):
        """_get_trace_duration_us() returns 0.0 for empty events."""
        prof = self._make_mock_prof(events=[])
        analyzer = OverlapAnalyzer()
        self.assertEqual(analyzer._get_trace_duration_us(prof), 0.0)

    def test_get_trace_duration_events_exception(self):
        """_get_trace_duration_us() returns 0.0 when events() raises."""
        prof = MagicMock(spec=torch.profiler.profile)
        prof.events.side_effect = RuntimeError("no trace")
        analyzer = OverlapAnalyzer()
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
        self.assertFalse(cfg.experimental_diagnostics)

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

    def test_experimental_diagnostics_adds_overlap_analyzer(self):
        """experimental_diagnostics=True adds OverlapAnalyzer automatically."""
        profiler = Profiler(Profiler.Config(experimental_diagnostics=True))
        self.assertEqual(len(profiler._analyzers), 1)
        self.assertIsInstance(profiler._analyzers[0], OverlapAnalyzer)

    def test_custom_analyzers_are_appended(self):
        """Custom analyzers passed via analyzers= are stored."""
        class MyAnalyzer(ProfileAnalyzer):
            def analyze(self, prof):
                pass

        my = MyAnalyzer()
        profiler = Profiler(Profiler.Config(), analyzers=[my])
        self.assertEqual(profiler._analyzers, [my])

    def test_diagnostics_and_custom_analyzers_combined(self):
        """experimental_diagnostics adds OverlapAnalyzer first; custom ones follow."""
        class MyAnalyzer(ProfileAnalyzer):
            def analyze(self, prof):
                pass

        my = MyAnalyzer()
        profiler = Profiler(
            Profiler.Config(experimental_diagnostics=True), analyzers=[my]
        )
        self.assertEqual(len(profiler._analyzers), 2)
        self.assertIsInstance(profiler._analyzers[0], OverlapAnalyzer)
        self.assertIs(profiler._analyzers[1], my)


class TestProfilerDisabledPaths(unittest.TestCase):
    """Tests for the no-op paths that require no GPU."""

    def test_enable_profiling_disabled_yields_none(self):
        profiler = Profiler(Profiler.Config(enable_profiling=False))
        with profiler.enable_profiling(
            global_step=0, base_folder="/tmp", leaf_folder=""
        ) as handle:
            self.assertIsNone(handle)

    def test_enable_memory_snapshot_disabled_yields_none(self):
        profiler = Profiler(Profiler.Config(enable_memory_snapshot=False))
        with profiler.enable_memory_snapshot(
            global_step=0, base_folder="/tmp", leaf_folder=""
        ) as handle:
            self.assertIsNone(handle)

    def test_enable_profiling_disabled_default_args(self):
        """enable_profiling works with all default keyword args."""
        profiler = Profiler(Profiler.Config())
        with profiler.enable_profiling() as handle:
            self.assertIsNone(handle)

    def test_enable_memory_snapshot_disabled_default_args(self):
        """enable_memory_snapshot works with all default keyword args."""
        profiler = Profiler(Profiler.Config())
        with profiler.enable_memory_snapshot() as handle:
            self.assertIsNone(handle)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestProfilerEnabledPathsGPU(unittest.TestCase):
    """GPU-dependent tests — skipped automatically when no GPU is present."""

    def test_enable_profiling_yields_profiler_handle(self):
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
            with profiler.enable_profiling(
                global_step=0, base_folder=tmpdir
            ) as handle:
                self.assertIsNotNone(handle)


if __name__ == "__main__":
    unittest.main()
