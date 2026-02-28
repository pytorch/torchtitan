# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.tools.profiler import Profiler


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
