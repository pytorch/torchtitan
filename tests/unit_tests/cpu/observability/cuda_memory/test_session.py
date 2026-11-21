# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import builtins
import gc
import importlib.util
import os
import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torchtitan

from torchtitan.observability.cuda_memory import session
from torchtitan.observability.profiler import (
    CudaMemoryProfilerConfig,
    CuspyProfilerConfig,
    Profiler,
    record_cuda_memory,
)


class TestCudaMemorySession(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = directory.name
        self.capture = MagicMock()
        self.capture.coverage = {}
        self.capture_factory = self.enterContext(
            patch.object(session, "MemoryCapture", return_value=self.capture)
        )
        self.set_device = self.enterContext(
            patch("torchtitan.observability.profiler.torch.cuda.set_device")
        )
        self.enterContext(
            patch("torchtitan.observability.profiler.get_local_device", return_value=1)
        )
        self.enterContext(patch.dict(os.environ, {"RANK": "3", "LOCAL_RANK": "1"}))

    def config(self, **kwargs):
        return Profiler.Config(cuda_memory=CudaMemoryProfilerConfig(**kwargs))

    def test_disabled_has_no_collector_import_or_cuda_work(self):
        original_import = builtins.__import__

        def check_import(name, *args, **kwargs):
            self.assertNotIn("cuda_memory", name)
            return original_import(name, *args, **kwargs)

        config = Profiler.Config()
        with patch.object(builtins, "__import__", side_effect=check_import):
            with record_cuda_memory(config, base_folder=self.output):
                with config.build() as profiler:
                    profiler.step()
        self.capture_factory.assert_not_called()
        self.set_device.assert_not_called()

    def test_startup_precedes_body_and_steps_bound_capture(self):
        config = self.config(capture_steps=2, stacks=False, max_records=4096)
        with record_cuda_memory(config, base_folder=self.output):
            self.capture.start.assert_called_once()
            self.set_device.assert_called_once_with(1)
            with config.build(global_step=100) as profiler:
                profiler.step()
                self.capture.stop.assert_not_called()
                profiler.step()
                self.capture.stop.assert_called_once()
                profiler.step()
        self.capture.stop.assert_called_once()
        self.capture_factory.assert_called_once_with(stacks=False, max_records=4096)
        destination = Path(self.output) / "profiling/traces/cuda_memory/rank3"
        self.capture.export.assert_called_once_with(destination / "memory.json.gz")
        self.capture.export_chrome_trace.assert_called_once_with(
            destination / "memory-trace.json.gz"
        )
        self.assertEqual(self.capture.coverage["completed_steps"], 2)
        self.assertEqual(self.capture.coverage["stop_reason"], "step_limit")

    def test_short_run_exports_on_scope_exit(self):
        config = self.config(capture_steps=3)
        with record_cuda_memory(config, base_folder=self.output):
            with config.build() as profiler:
                profiler.step()
        self.assertEqual(self.capture.coverage["completed_steps"], 1)
        self.assertEqual(self.capture.coverage["stop_reason"], "scope_exit")

    def test_window_releases_capture_before_training_scope_exits(self):
        references = []

        class Capture:
            def __init__(self, **kwargs):
                self.coverage = {}
                references.append(weakref.ref(self))

            def start(self):
                pass

            def stop(self):
                pass

            def export(self, path):
                pass

            def export_chrome_trace(self, path):
                pass

        self.capture_factory.side_effect = Capture
        config = self.config(capture_steps=1)
        with record_cuda_memory(config, base_folder=self.output):
            with config.build() as profiler:
                profiler.step()
                gc.collect()
                self.assertIsNone(references[0]())

    def test_initialization_failure_stops_without_success_export(self):
        failure = RuntimeError("trainer construction failed")
        with self.assertRaises(RuntimeError) as caught:
            with record_cuda_memory(self.config(), base_folder=self.output):
                raise failure
        self.assertIs(caught.exception, failure)
        self.capture.stop.assert_called_once()
        self.capture.export.assert_not_called()
        # A failed scope must not leave its context installed.
        with record_cuda_memory(self.config(), base_folder=self.output + "/retry"):
            pass

    def test_cleanup_failure_preserves_training_error(self):
        self.capture.stop.side_effect = RuntimeError("stop failed")
        failure = RuntimeError("training failed")
        with self.assertRaises(RuntimeError) as caught:
            with record_cuda_memory(self.config(), base_folder=self.output):
                raise failure
        self.assertIs(caught.exception, failure)
        self.assertIn("stop failed", failure.__notes__[0])

    def test_step_failure_stops_once_and_never_exports(self):
        self.capture.stop.side_effect = RuntimeError("overflow")
        config = self.config(capture_steps=1)
        with self.assertRaisesRegex(RuntimeError, "overflow"):
            with record_cuda_memory(config, base_folder=self.output):
                with config.build() as profiler:
                    profiler.step()
        self.capture.stop.assert_called_once()
        self.capture.export.assert_not_called()

    def test_start_failure_does_not_leave_scope_installed(self):
        self.capture.start.side_effect = RuntimeError("registration failed")
        with self.assertRaisesRegex(RuntimeError, "registration failed"):
            with record_cuda_memory(self.config(), base_folder=self.output):
                self.fail("scope entered after start failure")
        self.capture.start.side_effect = None
        with record_cuda_memory(self.config(), base_folder=self.output):
            pass

    def test_existing_evidence_is_not_overwritten(self):
        directory = Path(self.output) / "profiling/traces/cuda_memory/rank3"
        directory.mkdir(parents=True)
        sentinel = directory / "memory.json.gz"
        sentinel.write_bytes(b"previous capture")
        with self.assertRaises(FileExistsError):
            with record_cuda_memory(self.config(), base_folder=self.output):
                pass
        self.assertEqual(sentinel.read_bytes(), b"previous capture")
        self.capture.start.assert_not_called()

    def test_rejects_kineto_before_start(self):
        config = self.config()
        config.enable_profiling = True
        with self.assertRaisesRegex(ValueError, "Kineto"):
            with record_cuda_memory(config, base_folder=self.output):
                pass
        self.capture.start.assert_not_called()
        self.set_device.assert_not_called()

    def test_cuspy_schedule_can_share_startup_scope(self):
        config = self.config()
        config.enable_profiling = True
        config.cuspy = CuspyProfilerConfig()
        scheduled = MagicMock()
        with patch.object(Profiler, "build_torch_profiler", return_value=scheduled):
            with record_cuda_memory(config, base_folder=self.output):
                with config.build() as profiler:
                    profiler.step()
        scheduled.step.assert_called_once()
        scheduled.__exit__.assert_called_once()

    def test_profiler_requires_early_scope(self):
        with self.assertRaisesRegex(RuntimeError, "trainer construction"):
            with self.config().build():
                pass

    def test_unrelated_profiler_does_not_advance_window(self):
        with record_cuda_memory(self.config(capture_steps=1), base_folder=self.output):
            with Profiler.Config().build() as profiler:
                profiler.step()
            self.capture.stop.assert_not_called()

    def test_nested_scope_rejected(self):
        with record_cuda_memory(self.config(), base_folder=self.output):
            with self.assertRaisesRegex(RuntimeError, "nested"):
                with record_cuda_memory(
                    self.config(), base_folder=self.output + "/nested"
                ):
                    pass

    def test_invalid_config_and_mutated_recipe_rejected(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                CudaMemoryProfilerConfig(capture_steps=value)
        for value in (0, 1048577, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                CudaMemoryProfilerConfig(max_records=value)
        config = self.config()
        config.cuda_memory.capture_steps = 0
        with self.assertRaises(ValueError):
            with record_cuda_memory(config, base_folder=self.output):
                pass

    def training_entry(self):
        # Exercise the real CLI lifecycle without importing model/data dependencies.
        spec = importlib.util.spec_from_file_location(
            "cuda_memory_training_entry", Path(torchtitan.__file__).parent / "train.py"
        )
        module = importlib.util.module_from_spec(spec)
        with patch.dict(
            sys.modules,
            {
                "torchtitan.trainer": SimpleNamespace(
                    Trainer=SimpleNamespace(Config=object)
                )
            },
        ):
            spec.loader.exec_module(module)
        return module

    def test_normal_entry_captures_constructor_training_and_teardown(self):
        entry = self.training_entry()
        trainer = MagicMock()
        config = SimpleNamespace(
            profiler=self.config(capture_steps=1),
            dump_folder=self.output,
            debug=SimpleNamespace(enable_structured_logging=False),
            create_seed_checkpoint=False,
        )

        def build():
            self.capture.start.assert_called_once()
            return trainer

        def train():
            with config.profiler.build() as profiler:
                profiler.step()

        config.build = build
        trainer.train.side_effect = train
        with (
            patch.object(entry.ConfigLoader, "load", return_value=config),
            patch.object(entry, "init_logger"),
            patch.object(entry.sl, "init_structured_logger"),
            patch.object(entry.torch.distributed, "is_initialized", return_value=True),
            patch.object(entry.torch.distributed, "destroy_process_group") as destroy,
        ):
            entry.main()
        self.capture.stop.assert_called_once()
        trainer.close.assert_called_once()
        destroy.assert_called_once()

    def test_normal_entry_constructor_failure_closes_capture(self):
        entry = self.training_entry()
        failure = RuntimeError("constructor failed")
        config = SimpleNamespace(
            profiler=self.config(),
            dump_folder=self.output,
            debug=SimpleNamespace(enable_structured_logging=False),
            build=MagicMock(side_effect=failure),
        )
        with (
            patch.object(entry.ConfigLoader, "load", return_value=config),
            patch.object(entry, "init_logger"),
            patch.object(entry.sl, "init_structured_logger"),
            self.assertRaises(RuntimeError) as caught,
        ):
            entry.main()
        self.assertIs(caught.exception, failure)
        self.capture.stop.assert_called_once()
        self.capture.export.assert_not_called()


if __name__ == "__main__":
    unittest.main()
