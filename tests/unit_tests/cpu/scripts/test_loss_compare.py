# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import loss_compare


def _git(repo: str, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


class TestLossCompareCommitResolution(unittest.TestCase):
    """The test scenario must run at the test commit, not at the baseline."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = self._tmp.name
        _git(self.repo, "init", "-q", "-b", "main")
        _git(self.repo, "config", "user.email", "test@example.com")
        _git(self.repo, "config", "user.name", "test")
        self.shas = []
        for version in ("old", "new"):
            with open(os.path.join(self.repo, "VERSION"), "w") as f:
                f.write(version)
            _git(self.repo, "add", "VERSION")
            _git(self.repo, "-c", "commit.gpgsign=false", "commit", "-q", "-m", version)
            self.shas.append(_git(self.repo, "rev-parse", "HEAD"))
        self._cwd = os.getcwd()
        os.chdir(self.repo)

    def tearDown(self):
        os.chdir(self._cwd)
        self._tmp.cleanup()

    def _run(self, baseline_commit: str, test_commit: str) -> dict[str, str]:
        """Run main() and return the commit each scenario trained at."""
        trained_at = {}

        def fake_run_training(scenario, *args, **kwargs):
            trained_at[scenario] = _git(self.repo, "rev-parse", "HEAD")
            return f"{scenario}.log"

        argv = [
            "loss_compare.py",
            baseline_commit,
            test_commit,
            "--no-seed-checkpoint",
            "--steps=2",
        ]
        with patch.object(sys, "argv", argv), patch.object(
            loss_compare, "run_training", side_effect=fake_run_training
        ), patch.object(
            loss_compare,
            "extract_metrics_from_tensorboard",
            return_value={"loss": {1: 1.0, 2: 1.0}},
        ):
            loss_compare.main()
        return trained_at

    def test_relative_test_commit_is_resolved_before_checkout(self):
        old, new = self.shas
        trained_at = self._run("HEAD~1", "HEAD")
        self.assertEqual(trained_at, {"baseline": old, "test": new})

    def test_dot_test_commit_uses_original_checkout(self):
        old, new = self.shas
        trained_at = self._run(old, ".")
        self.assertEqual(trained_at, {"baseline": old, "test": new})
        # The original branch is restored afterwards.
        self.assertEqual(_git(self.repo, "rev-parse", "--abbrev-ref", "HEAD"), "main")

    def test_unknown_commit_exits_before_checkout(self):
        with self.assertRaises(SystemExit):
            self._run("does-not-exist", "HEAD")
        self.assertEqual(_git(self.repo, "rev-parse", "--abbrev-ref", "HEAD"), "main")


class TestLossCompareTensorBoard(unittest.TestCase):
    def test_retains_all_recorded_scalar_steps(self):
        from tensorboard.compat.proto import event_pb2, summary_pb2
        from tensorboard.summary.writer.event_file_writer import EventFileWriter

        num_steps = 10_001
        with tempfile.TemporaryDirectory() as dump_folder:
            event_dir = os.path.join(dump_folder, "tb", "run")
            writer = EventFileWriter(event_dir)
            try:
                for step in range(num_steps):
                    writer.add_event(
                        event_pb2.Event(
                            step=step,
                            summary=summary_pb2.Summary(
                                value=[
                                    summary_pb2.Summary.Value(
                                        tag=tag, simple_value=float(step)
                                    )
                                    for tag in loss_compare.TB_TAGS.values()
                                ]
                            ),
                        )
                    )
            finally:
                writer.close()

            metrics = loss_compare.extract_metrics_from_tensorboard(
                dump_folder, "tb", tuple(loss_compare.TB_TAGS)
            )

            expected = {step: float(step) for step in range(num_steps)}
            for name, values in metrics.items():
                with self.subTest(metric=name):
                    self.assertEqual(len(values), num_steps)
                    self.assertEqual(values, expected)


if __name__ == "__main__":
    unittest.main()
