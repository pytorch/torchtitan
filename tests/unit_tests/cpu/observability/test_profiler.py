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
        self.assertIsNone(cfg.profile_ranks)

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

    def test_neuron_adds_privateuse1_activity_and_stacks(self):
        import tempfile

        import torch

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch("torch.xpu.is_available", return_value=False),
            mock.patch("torchtitan.observability.profiler.device_type", "neuron"),
            mock.patch("torch.profiler.profile") as profile,
        ):
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
            profiler.build_torch_profiler(
                global_step=0, base_folder=tmpdir, leaf_folder=""
            )
        self.assertEqual(
            profile.call_args.kwargs["activities"],
            [
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.PrivateUse1,
            ],
        )
        self.assertTrue(profile.call_args.kwargs["with_stack"])

    def test_profile_ranks_none_profiles_every_rank(self):
        """The default (None) keeps every rank profiling."""
        import tempfile

        for rank in (0, 7):
            with mock.patch("torch.distributed.get_rank", return_value=rank):
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

    def test_profile_ranks_selects_listed_ranks(self):
        """A rank in profile_ranks still builds a profiler."""
        import tempfile

        with mock.patch("torch.distributed.get_rank", return_value=3):
            with tempfile.TemporaryDirectory() as tmpdir:
                profiler = Profiler(
                    Profiler.Config(
                        enable_profiling=True,
                        profile_freq=4,
                        profiler_warmup=1,
                        profiler_active=1,
                        profile_ranks=[0, 3],
                    ),
                    global_step=0,
                    base_folder=tmpdir,
                )
                with profiler:
                    self.assertIsNotNone(profiler.torch_profiler)

    def test_profile_ranks_unlisted_rank_still_traces(self):
        """An unlisted rank still traces, so profiling cost stays balanced,
        but does not create the trace directory."""
        import os
        import tempfile

        with mock.patch("torch.distributed.get_rank", return_value=5):
            with tempfile.TemporaryDirectory() as tmpdir:
                profiler = Profiler(
                    Profiler.Config(
                        enable_profiling=True,
                        profile_freq=4,
                        profiler_warmup=1,
                        profiler_active=1,
                        profile_ranks=[0],
                    ),
                    global_step=0,
                    base_folder=tmpdir,
                )
                with profiler:
                    self.assertIsNotNone(profiler.torch_profiler)
                    self.assertEqual(os.listdir(tmpdir), [])

    def _run_export(self, rank, profile_ranks, tmpdir):
        """Step a profiler through one active window so its trace is exported."""
        with mock.patch("torch.distributed.get_rank", return_value=rank):
            profiler = Profiler(
                Profiler.Config(
                    enable_profiling=True,
                    profile_freq=3,
                    profiler_warmup=1,
                    profiler_active=1,
                    profile_ranks=profile_ranks,
                ),
                global_step=0,
                base_folder=tmpdir,
            )
            with profiler:
                for _ in range(3):
                    torch.ones(4).add_(1)
                    profiler.step()

    def test_profile_ranks_listed_rank_saves_gzipped_trace(self):
        """A listed rank writes a readable gzipped chrome trace."""
        import glob
        import gzip
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            self._run_export(rank=3, profile_ranks=[0, 3], tmpdir=tmpdir)
            files = glob.glob(
                os.path.join(tmpdir, "profiling/traces/*/rank3_trace.json.gz")
            )
            self.assertEqual(len(files), 1)
            with gzip.open(files[0], "rt") as f:
                self.assertIn("traceEvents", json.load(f))

    def test_profile_ranks_unlisted_rank_gzips_to_devnull(self):
        """An unlisted rank pays the same export and gzip cost, writing the
        result to os.devnull and leaving nothing on disk."""
        import gzip
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch(
                "torchtitan.observability.profiler.gzip.open", wraps=gzip.open
            ) as gzip_open:
                self._run_export(rank=5, profile_ranks=[0], tmpdir=tmpdir)
            gzip_open.assert_called_once_with(os.devnull, "wb")
            self.assertEqual(os.listdir(tmpdir), [])

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


if __name__ == "__main__":
    unittest.main()
