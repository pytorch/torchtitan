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

OUT_OF_ORDER_RUNNER = """
import json, os, sys, time

args = sys.argv[1:]
step = int(args[args.index("--step") + 1])
release_path = args[args.index("--release-path") + 1]
if step == 10:
    deadline = time.monotonic() + 10
    while not os.path.exists(release_path):
        if time.monotonic() >= deadline:
            raise TimeoutError("step 10 was not released")
        time.sleep(0.01)
with open(args[args.index("--result-path") + 1], "w") as f:
    json.dump({"step": step, "metrics": {"loss": 1.5}}, f)
"""


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
            freq=overrides.pop("freq", 1),
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
        checkpoint_dir = os.path.join(self.dump_folder, f"step-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        async_eval.async_eval_launch_command(step, checkpoint_dir)
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
        self.assertFalse(async_eval.request_async_eval_steps(100))
        self.assertFalse(config.enable)
        self.assertIsInstance(async_eval.logger, BaseLogger)

    def test_default_runner_is_example_loss_eval_module(self):
        config = AsyncEval.Config()
        self.assertEqual(
            config.runner,
            "-m torchtitan_recipes.async_eval.example_loss_eval",
        )

    def test_empty_runner_when_enabled(self):
        with self.assertRaisesRegex(ValueError, "runner cannot be empty"):
            AsyncEval.Config(enable=True, runner=" ")

    def test_freq_must_be_positive(self):
        with self.assertRaisesRegex(
            ValueError, "freq must be at least 1"
        ):
            AsyncEval.Config(freq=0)

    def test_checkpoint_trigger_and_retention_exemption(self):
        async_eval = self.build_async_eval(freq=2)

        self.assertFalse(async_eval.request_async_eval_steps(1))
        self.assertTrue(async_eval.request_async_eval_steps(2))

        with mock.patch.object(async_eval, "_launch") as launch:
            async_eval.async_eval_launch_command(1, "step-1")
            launch.assert_not_called()
            async_eval.async_eval_launch_command(2, "step-2")
            launch.assert_called_once_with(2, "step-2")

        async_eval.config.preserve_checkpoints = False
        self.assertTrue(async_eval.request_async_eval_steps(2))

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

    def test_preserve_false_keeps_only_latest_eval_checkpoint(self):
        async_eval = self.build_async_eval(preserve_checkpoints=False)
        first_checkpoint = self.report_completed_save(async_eval, 10)
        async_eval.jobs[0].process.wait()
        async_eval.collect()
        self.assertTrue(os.path.isdir(first_checkpoint))

        second_checkpoint = self.report_completed_save(async_eval, 20)
        async_eval.jobs[0].process.wait()
        self.assertTrue(os.path.isdir(first_checkpoint))
        async_eval.collect()

        self.assertFalse(os.path.exists(first_checkpoint))
        self.assertTrue(os.path.isdir(second_checkpoint))
        async_eval.close()

    def test_eval_checkpoints_can_finish_out_of_order(self):
        release_path = os.path.join(self.dump_folder, "release-step-10")
        async_eval = self.build_async_eval(
            OUT_OF_ORDER_RUNNER,
            preserve_checkpoints=False,
            extra_args=f"--release-path {release_path}",
        )
        first_checkpoint = self.report_completed_save(async_eval, 10)
        second_checkpoint = self.report_completed_save(async_eval, 20)
        jobs_by_step = {job.step: job for job in async_eval.jobs}

        jobs_by_step[20].process.wait(timeout=5)
        async_eval.collect()
        self.assertTrue(os.path.isdir(first_checkpoint))
        self.assertTrue(os.path.isdir(second_checkpoint))

        with open(release_path, "w"):
            pass
        jobs_by_step[10].process.wait(timeout=5)
        async_eval.collect()

        self.assertFalse(os.path.exists(first_checkpoint))
        self.assertTrue(os.path.isdir(second_checkpoint))
        async_eval.close()

    def test_preserve_true_keeps_all_eval_checkpoints(self):
        async_eval = self.build_async_eval(preserve_checkpoints=True)
        first_checkpoint = self.report_completed_save(async_eval, 10)
        async_eval.jobs[0].process.wait()
        async_eval.collect()
        second_checkpoint = self.report_completed_save(async_eval, 20)
        async_eval.close()

        self.assertTrue(os.path.isdir(first_checkpoint))
        self.assertTrue(os.path.isdir(second_checkpoint))

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


class TestAsyncEvalFailures(AsyncEvalTestCase):
    def test_failing_runner_does_not_stop_training(self):
        async_eval = self.build_async_eval(FAILING_RUNNER)
        self.report_completed_save(async_eval, 10)

        logger = self.run_to_completion(async_eval)
        self.assertEqual(logger.records, [])

    def test_failing_runner_can_be_made_fatal(self):
        async_eval = self.build_async_eval(
            FAILING_RUNNER,
            raise_on_failure=True,
            preserve_checkpoints=False,
        )
        first_checkpoint = self.report_completed_save(async_eval, 10)
        second_checkpoint = self.report_completed_save(async_eval, 20)
        for job in async_eval.jobs:
            job.process.wait()

        with self.assertRaisesRegex(RuntimeError, "failed with exit code 1"):
            async_eval.collect()

        self.assertEqual(async_eval.jobs, [])
        self.assertFalse(os.path.exists(first_checkpoint))
        self.assertTrue(os.path.isdir(second_checkpoint))
        # The failed jobs are not reported twice.
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
        checkpoint_dir = self.report_completed_save(async_eval, 10)

        async_eval.close()
        self.assertEqual(len(async_eval.jobs), 1)
        self.assertTrue(os.path.isdir(checkpoint_dir))
        async_eval.jobs[0].process.kill()
        async_eval.jobs[0].process.wait()


if __name__ == "__main__":
    unittest.main()
