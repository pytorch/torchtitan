# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest import mock

import torch

from torchtitan.observability.profiler import Profiler


class TestProfilerConfig(unittest.TestCase):
    def test_default_field_values(self):
        cfg = Profiler.Config()
        self.assertFalse(cfg.enable_profiling)
        self.assertEqual(cfg.save_traces_folder, "profiling/traces")
        self.assertEqual(cfg.profile_freq, 10)
        self.assertEqual(cfg.profiler_active, 1)
        self.assertEqual(cfg.profiler_warmup, 3)
        self.assertIsNone(cfg.profiler_repeat)
        self.assertIsNone(cfg.profiler_skip_first)
        self.assertIsNone(cfg.profiler_skip_first_wait)
        self.assertFalse(cfg.enable_memory_snapshot)
        self.assertEqual(cfg.save_memory_snapshot_folder, "profiling/memory_snapshot")
        self.assertIsNone(cfg.memory_snapshot_freq)

    def test_custom_field_values(self):
        cfg = Profiler.Config(
            enable_profiling=True,
            save_traces_folder="my_traces",
            profile_freq=50,
            profiler_repeat=2,
            profiler_skip_first=5,
            profiler_skip_first_wait=3,
            memory_snapshot_freq=7,
        )
        self.assertTrue(cfg.enable_profiling)
        self.assertEqual(cfg.save_traces_folder, "my_traces")
        self.assertEqual(cfg.profile_freq, 50)
        self.assertEqual(cfg.profiler_repeat, 2)
        self.assertEqual(cfg.profiler_skip_first, 5)
        self.assertEqual(cfg.profiler_skip_first_wait, 3)
        self.assertEqual(cfg.memory_snapshot_freq, 7)

    def test_build_returns_profiler_instance(self):
        """Profiler.Config.build() auto-wires to Profiler via Configurable."""
        cfg = Profiler.Config()
        profiler = cfg.build()
        self.assertIsInstance(profiler, Profiler)


class TestProfilerInit(unittest.TestCase):
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

    def test_active_updates_runtime_args(self):
        """active() updates runtime args and returns self for context manager use."""
        profiler = Profiler(Profiler.Config())
        self.assertEqual(profiler._global_step, 0)
        self.assertEqual(profiler._base_folder, "")
        self.assertEqual(profiler._leaf_folder, "")

        result = profiler.active(
            global_step=10, base_folder="/output", leaf_folder="replica_0"
        )
        self.assertIs(result, profiler)
        self.assertEqual(profiler._global_step, 10)
        self.assertEqual(profiler._base_folder, "/output")
        self.assertEqual(profiler._leaf_folder, "replica_0")

    def test_active_as_context_manager(self):
        """active() can be used as a context manager with 'with' statement."""
        profiler = Profiler(Profiler.Config())
        with profiler.active(global_step=5, base_folder="/tmp") as prof:
            self.assertIs(prof, profiler)
            self.assertEqual(prof._global_step, 5)
            prof.step()


class TestProfilerEnabledPaths(unittest.TestCase):
    """Tests for enabled profiler paths — uses mocked distributed rank."""

    def setUp(self):
        self.patcher_rank = mock.patch("torch.distributed.get_rank", return_value=0)
        self.patcher_rank.start()

    def tearDown(self):
        self.patcher_rank.stop()

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

    def test_memory_snapshot_frequency_is_independent(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch(
                "torchtitan.observability.profiler.MemoryProfiler"
            ) as memory_profiler_cls:
                profiler = Profiler(
                    Profiler.Config(
                        enable_memory_snapshot=True,
                        profile_freq=50,
                        memory_snapshot_freq=3,
                    )
                )
                memory_profiler = profiler.build_memory_profiler(
                    global_step=7,
                    base_folder=tmpdir,
                    leaf_folder="",
                )

        self.assertIs(memory_profiler, memory_profiler_cls.return_value)
        self.assertEqual(memory_profiler_cls.call_args.args[1], 3)

    def test_memory_snapshot_frequency_defaults_to_profile_frequency(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch(
                "torchtitan.observability.profiler.MemoryProfiler"
            ) as memory_profiler_cls:
                profiler = Profiler(
                    Profiler.Config(
                        enable_memory_snapshot=True,
                        profile_freq=6,
                    )
                )
                profiler.build_memory_profiler(
                    global_step=0,
                    base_folder=tmpdir,
                    leaf_folder="",
                )

        self.assertEqual(memory_profiler_cls.call_args.args[1], 6)

    def test_memory_snapshot_frequency_must_be_positive(self):
        import tempfile

        profiler = Profiler(
            Profiler.Config(
                enable_memory_snapshot=True,
                memory_snapshot_freq=0,
            )
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(
                ValueError, "Memory snapshot frequency must be greater than zero"
            ):
                profiler.build_memory_profiler(
                    global_step=0,
                    base_folder=tmpdir,
                    leaf_folder="",
                )

    def test_cuspy_raises_when_environment_unsupported(self):
        import tempfile

        from torchtitan.observability.profiler import CuspyProfilerConfig

        profiler = Profiler(
            Profiler.Config(
                enable_profiling=True,
                profile_freq=4,
                profiler_warmup=1,
                profiler_active=1,
                cuspy=CuspyProfilerConfig(),
            )
        )
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch(
                "torchtitan.observability.profiler._cuspy_unavailable_reasons",
                return_value=["reason one", "reason two"],
            ),
        ):
            with self.assertRaisesRegex(
                ValueError, "Cuspy cannot run here:\n  - reason one\n  - reason two"
            ):
                profiler.build_torch_profiler(
                    global_step=0, base_folder=tmpdir, leaf_folder=""
                )

    def _build_cuspy_profiler(self, tmpdir, cuspy):
        """Build a Cuspy-configured profiler with the environment checks and the
        torch profiler mocked out; return the ``torch.profiler.profile`` mock."""
        profiler = Profiler(
            Profiler.Config(
                enable_profiling=True,
                profile_freq=4,
                profiler_warmup=1,
                profiler_active=1,
                cuspy=cuspy,
            )
        )
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch(
                "torchtitan.observability.profiler._cuspy_unavailable_reasons",
                return_value=[],
            ),
            mock.patch(
                "torchtitan.observability.profiler._cuspy_activity_config",
                return_value="cuspy-activity",
            ),
            mock.patch("torch.profiler.profile") as profile,
        ):
            profiler.build_torch_profiler(
                global_step=0, base_folder=tmpdir, leaf_folder=""
            )
        return profile

    def test_cuspy_trace_file_name_follows_format(self):
        import os
        import tempfile

        from torchtitan.observability.profiler import CuspyProfilerConfig

        for trace_format, file_name in (
            ("json", "rank0_trace.json.gz"),
            ("pftrace", "rank0_trace.pftrace.gz"),
        ):
            with self.subTest(trace_format=trace_format):
                with tempfile.TemporaryDirectory() as tmpdir:
                    profile = self._build_cuspy_profiler(
                        tmpdir, CuspyProfilerConfig(trace_format=trace_format)
                    )
                    kwargs = profile.call_args.kwargs
                    self.assertIn("cuspy-activity", kwargs["activities"])
                    prof = mock.Mock(step_num=4)
                    kwargs["on_trace_ready"](prof)
                    prof.export_chrome_trace.assert_called_once_with(
                        os.path.join(
                            tmpdir, "profiling/traces", "iteration_4", file_name
                        )
                    )

    def test_cuspy_requires_cuda(self):
        import tempfile

        from torchtitan.observability.profiler import CuspyProfilerConfig

        profiler = Profiler(
            Profiler.Config(enable_profiling=True, cuspy=CuspyProfilerConfig())
        )
        for xpu_available in (True, False):
            with (
                self.subTest(xpu_available=xpu_available),
                tempfile.TemporaryDirectory() as tmpdir,
                mock.patch("torch.cuda.is_available", return_value=False),
                mock.patch("torch.xpu.is_available", return_value=xpu_available),
                self.assertRaisesRegex(ValueError, "profiler.cuspy requires CUDA"),
            ):
                profiler.build_torch_profiler(
                    global_step=0, base_folder=tmpdir, leaf_folder=""
                )

    def test_cuspy_pm_sampling_options_require_metrics(self):
        from torchtitan.observability.profiler import CuspyProfilerConfig

        with self.assertRaisesRegex(ValueError, "require performance_metrics"):
            CuspyProfilerConfig(pm_lookback_window_ms=60_000.0)

    @unittest.skipUnless(
        hasattr(torch.profiler, "CuspyConfig"), "torch build has no Cuspy"
    )
    def test_cuspy_activity_config(self):
        from torchtitan.observability.profiler import (
            _cuspy_activity_config,
            CuspyProfilerConfig,
        )

        activity_config = _cuspy_activity_config(
            CuspyProfilerConfig(enable_event_node_ids=True)
        )[torch.profiler.ProfilerActivity.CUDA]
        (cuspy_config,) = activity_config.profiler_configs
        self.assertIsInstance(cuspy_config, torch.profiler.CuspyConfig)
        self.assertTrue(cuspy_config.enable_event_node_ids)
        self.assertFalse(cuspy_config.enable_cuda_sync_events)

        activity_config = _cuspy_activity_config(
            CuspyProfilerConfig(
                performance_metrics=["sm__cycles_active.avg"],
                pm_sampling_interval_ms=0.5,
                pm_lookback_window_ms=60_000.0,
            )
        )[torch.profiler.ProfilerActivity.CUDA]
        _, pm_config = activity_config.profiler_configs
        self.assertEqual(pm_config.metric_names, ["sm__cycles_active.avg"])
        self.assertEqual(pm_config.sampling_interval_ms, 0.5)
        self.assertEqual(pm_config.lookback_window_ms, 60_000.0)

    @unittest.skipUnless(
        hasattr(torch.profiler, "CuspyConfig"), "torch build has no Cuspy"
    )
    def test_cuspy_pftrace_requires_native_encoder(self):
        from torchtitan.observability.profiler import (
            _cuspy_unavailable_reasons,
            CuspyProfilerConfig,
        )

        with mock.patch.object(
            torch._C._profiler._cuspy,
            "encode_pftrace",
            side_effect=RuntimeError("built without native .pftrace support"),
        ):
            self.assertIn(
                'trace_format="pftrace" is unsupported: built without native '
                ".pftrace support",
                _cuspy_unavailable_reasons(CuspyProfilerConfig(trace_format="pftrace")),
            )
            self.assertNotIn(
                'trace_format="pftrace" is unsupported: built without native '
                ".pftrace support",
                _cuspy_unavailable_reasons(CuspyProfilerConfig()),
            )


if __name__ == "__main__":
    unittest.main()
