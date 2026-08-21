# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

from torchtitan.components.async_eval import AsyncEval
from torchtitan.components.metrics import BaseLogger, MetricsProcessor

# Stands in for a real eval runner: reports a loss and echoes the arguments it
# was launched with, so tests can check the runner contract.
FAKE_RUNNER = """
import json, sys

args = sys.argv[1:]
metrics = {"loss": 1.5, "task": "made-up"}
with open(args[args.index("--result-path") + 1], "w") as f:
    json.dump({"step": int(args[args.index("--step") + 1]),
               "metrics": metrics, "args": args}, f)
"""

FAILING_RUNNER = "import sys; sys.exit(1)"

SILENT_RUNNER = "pass"


class FakeCheckpointer:
    """Minimal checkpointer that can report completed saves."""

    def __init__(
        self,
        folder: str,
        enable: bool = True,
        load_only: bool = False,
        eval_freq: int = 10,
    ):
        self.load_only = load_only
        self.enable = enable
        self.folder = folder
        self.eval_freq = eval_freq
        self.callback = None

    def register_eval_checkpoint_callback(self, callback) -> None:
        self.callback = callback

    def complete_save(self, step: int) -> str:
        checkpoint_dir = os.path.join(self.folder, f"step-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        if self.callback is not None and step % self.eval_freq == 0:
            self.callback(step, checkpoint_dir)
        return checkpoint_dir


class RecordingLogger(BaseLogger):
    def __init__(self):
        self.records: list[tuple[int, dict]] = []
        self.closed = False

    def log(self, metrics: dict, step: int) -> None:
        self.records.append((step, metrics))

    def close(self) -> None:
        self.closed = True


class AsyncEvalTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.dump_folder = self.tmp_dir.name
        self.runner_path = os.path.join(self.dump_folder, "runner.py")

    def build_async_eval(self, runner_source: str = FAKE_RUNNER, **overrides):
        """An AsyncEval that runs a local script instead of a training job."""
        with open(self.runner_path, "w") as f:
            f.write(runner_source)

        config = AsyncEval.Config(
            enable=True,
            launcher="",
            runner=f"{sys.executable} {self.runner_path}",
            forward_train_args=overrides.pop("forward_train_args", False),
            exit_timeout=overrides.pop("exit_timeout", 30.0),
            **overrides,
        )
        async_eval = config.build(dump_folder=self.dump_folder)
        async_eval.logger = RecordingLogger()
        return async_eval

    def report_completed_save(self, async_eval: AsyncEval, step: int) -> str:
        checkpointer = FakeCheckpointer(self.dump_folder)
        async_eval.register_checkpoint_callback(checkpointer)
        checkpoint_dir = checkpointer.complete_save(step)
        async_eval.collect()
        return checkpoint_dir

    def run_to_completion(self, async_eval: AsyncEval) -> RecordingLogger:
        async_eval.close()
        self.assertEqual(async_eval.jobs, [], "eval job did not finish in time")
        return async_eval.logger

    def read_result(self, step: int) -> dict:
        result_path = os.path.join(
            self.dump_folder, "async_eval", f"step-{step}", "result.json"
        )
        with open(result_path) as f:
            return json.load(f)


class TestAsyncEvalConfig(AsyncEvalTestCase):
    def test_disabled_by_default(self):
        config = AsyncEval.Config()
        async_eval = config.build()
        self.assertFalse(config.enable)
        self.assertIsInstance(async_eval.logger, BaseLogger)

    def test_default_runner_is_example_loss_eval_module(self):
        config = AsyncEval.Config()
        self.assertEqual(
            config.runner,
            "-m torchtitan.components.async_eval.example_loss_eval",
        )

    def test_empty_runner_when_enabled(self):
        with self.assertRaisesRegex(ValueError, "runner cannot be empty"):
            AsyncEval.Config(enable=True, runner=" ")

    def test_tensorboard_logger_follows_metrics_config(self):
        config = AsyncEval.Config(enable=True)
        metrics_config = MetricsProcessor.Config(enable_tensorboard=True)
        with mock.patch(
            "torchtitan.components.async_eval.manager.TensorBoardLogger"
        ) as tb_logger:
            config.build(dump_folder=self.dump_folder, metrics_config=metrics_config)
        log_dir = tb_logger.call_args.args[0]
        self.assertTrue(
            log_dir.startswith(os.path.join(self.dump_folder, "tb", "async_eval")),
            log_dir,
        )


class TestAsyncEvalLaunch(AsyncEvalTestCase):
    def test_metrics_are_logged_at_the_launch_step(self):
        async_eval = self.build_async_eval()
        self.report_completed_save(async_eval, 10)

        logger = self.run_to_completion(async_eval)
        # The non-numeric metric the runner reported is dropped, not fatal.
        self.assertEqual(logger.records, [(10, {"loss": 1.5})])
        self.assertTrue(logger.closed)

    def test_runner_receives_the_contract_arguments(self):
        async_eval = self.build_async_eval(extra_args="--tasks mmlu")
        self.report_completed_save(async_eval, 20)
        self.run_to_completion(async_eval)

        args = self.read_result(20)["args"]
        step_dir = os.path.join(self.dump_folder, "async_eval", "step-20")
        self.assertEqual(
            args,
            [
                "--tasks",
                "mmlu",
                "--checkpoint-dir",
                os.path.join(self.dump_folder, "step-20"),
                "--output-dir",
                step_dir,
                "--step",
                "20",
                "--result-path",
                os.path.join(step_dir, "result.json"),
            ],
        )

    def test_extra_args_precede_train_args(self):
        # tyro rejects options that follow a subcommand, and a training command
        # line can end with one.
        with mock.patch.object(
            sys,
            "argv",
            ["train.py", "--module", "llama3", "activation-checkpoint:none"],
        ):
            async_eval = self.build_async_eval(
                forward_train_args=True, extra_args="--validator.steps 2"
            )

        self.report_completed_save(async_eval, 10)
        self.run_to_completion(async_eval)

        args = self.read_result(10)["args"]
        self.assertEqual(
            args[:5],
            [
                "--validator.steps",
                "2",
                "--module",
                "llama3",
                "activation-checkpoint:none",
            ],
        )

    def test_non_eval_checkpoint_does_not_launch(self):
        async_eval = self.build_async_eval()
        checkpointer = FakeCheckpointer(self.dump_folder)
        async_eval.register_checkpoint_callback(checkpointer)

        checkpointer.complete_save(5)
        async_eval.collect()

        self.assertEqual(async_eval.jobs, [])

    def test_registration_requires_checkpointing(self):
        async_eval = self.build_async_eval()
        checkpointer = FakeCheckpointer(self.dump_folder, enable=False)

        with self.assertRaisesRegex(ValueError, "checkpoint.enable=True"):
            async_eval.register_checkpoint_callback(checkpointer)

    def test_registration_requires_checkpoint_saves(self):
        async_eval = self.build_async_eval()
        checkpointer = FakeCheckpointer(self.dump_folder, load_only=True)

        with self.assertRaisesRegex(ValueError, "checkpoint.load_only=False"):
            async_eval.register_checkpoint_callback(checkpointer)

    def test_disabled_async_eval_does_not_launch(self):
        async_eval = self.build_async_eval()
        async_eval.config.enable = False

        checkpointer = FakeCheckpointer(self.dump_folder)
        async_eval.register_checkpoint_callback(checkpointer)
        checkpointer.complete_save(10)
        async_eval.collect()
        self.assertIsNone(checkpointer.callback)
        self.assertEqual(async_eval.jobs, [])


class TestAsyncEvalFailures(AsyncEvalTestCase):
    def test_failing_runner_does_not_stop_training(self):
        async_eval = self.build_async_eval(FAILING_RUNNER)
        self.report_completed_save(async_eval, 10)

        logger = self.run_to_completion(async_eval)
        self.assertEqual(logger.records, [])

    def test_failing_runner_can_be_made_fatal(self):
        async_eval = self.build_async_eval(FAILING_RUNNER, raise_on_failure=True)
        self.report_completed_save(async_eval, 10)

        with self.assertRaisesRegex(RuntimeError, "failed with exit code 1"):
            async_eval.close()
        # The failed job is not reported twice.
        async_eval.close()

    def test_runner_reporting_no_metrics_can_be_made_fatal(self):
        async_eval = self.build_async_eval(SILENT_RUNNER, raise_on_failure=True)
        self.report_completed_save(async_eval, 10)

        with self.assertRaisesRegex(RuntimeError, "reported no metrics"):
            async_eval.close()

    def test_missing_result_file(self):
        async_eval = self.build_async_eval(SILENT_RUNNER)
        self.report_completed_save(async_eval, 10)

        logger = self.run_to_completion(async_eval)
        self.assertEqual(logger.records, [])

    def test_malformed_result_file(self):
        async_eval = self.build_async_eval()
        result_path = os.path.join(
            self.dump_folder, "async_eval", "step-10", "result.json"
        )
        os.makedirs(os.path.dirname(result_path))
        job = mock.Mock(step=10, result_path=result_path)

        with open(job.result_path, "w") as f:
            f.write("not json")
        self.assertEqual(async_eval._read_metrics(job), {})

        with open(job.result_path, "w") as f:
            json.dump({"loss": 1.0}, f)
        self.assertEqual(async_eval._read_metrics(job), {})

    def test_unfinished_jobs_are_not_waited_for_forever(self):
        async_eval = self.build_async_eval(
            "import time; time.sleep(60)", exit_timeout=0.0
        )
        self.report_completed_save(async_eval, 10)

        async_eval.close()
        self.assertEqual(len(async_eval.jobs), 1)
        async_eval.jobs[0].process.kill()
        async_eval.jobs[0].process.wait()


if __name__ == "__main__":
    unittest.main()
