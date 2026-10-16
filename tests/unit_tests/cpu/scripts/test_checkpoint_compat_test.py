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

from scripts import checkpoint_compat_test


def _git(repo: str, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


class TestCheckpointCompatCommitResolution(unittest.TestCase):
    """Each run must happen at the commit its step names, including for "."."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = os.path.join(self._tmp.name, "repo")
        self.output_folder = os.path.join(self._tmp.name, "output")
        os.makedirs(self.repo)
        _git(self.repo, "init", "-q", "-b", "main")
        _git(self.repo, "config", "user.email", "test@example.com")
        _git(self.repo, "config", "user.name", "test")
        self.shas = []
        for version in ("old", "new"):
            with open(os.path.join(self.repo, "VERSION"), "w") as f:
                f.write(version)
            _git(self.repo, "add", "VERSION")
            _git(self.repo, "-c", "commit.gpgsign=false", "commit", "-qm", version)
            self.shas.append(_git(self.repo, "rev-parse", "HEAD"))
        self._cwd = os.getcwd()
        os.chdir(self.repo)

    def tearDown(self):
        os.chdir(self._cwd)
        self._tmp.cleanup()

    def _run(self, save_commit: str, load_commit: str) -> dict[str, str]:
        """Run main() and return the commit each training run used."""
        ran_at = {}

        def fake_run_cmd(cmd, logfile, ngpus, env_overrides=None):
            run_name = os.path.splitext(os.path.basename(logfile))[0]
            ran_at[run_name] = _git(self.repo, "rev-parse", "HEAD")

        argv = [
            "checkpoint_compat_test.py",
            save_commit,
            load_commit,
            "--steps=4",
            "--resume-step=2",
            f"--output-folder={self.output_folder}",
        ]
        with patch.object(sys, "argv", argv), patch.object(
            checkpoint_compat_test, "run_cmd", side_effect=fake_run_cmd
        ), patch.object(
            checkpoint_compat_test, "extract_tb_losses", return_value={1: 1.0}
        ):
            checkpoint_compat_test.main()
        return ran_at

    def test_dot_load_commit_resumes_at_original_checkout(self):
        old, new = self.shas
        ran_at = self._run("HEAD~1", ".")
        self.assertEqual(ran_at, {"reference": new, "save": old, "resume": new})
        self.assertEqual(_git(self.repo, "rev-parse", "--abbrev-ref", "HEAD"), "main")

    def test_dot_save_commit_saves_at_original_checkout(self):
        old, new = self.shas
        ran_at = self._run(".", old)
        self.assertEqual(ran_at, {"reference": old, "save": new, "resume": old})

    def test_unknown_commit_exits_before_checkout(self):
        with self.assertRaises(SystemExit):
            self._run("does-not-exist", "HEAD")
        self.assertEqual(_git(self.repo, "rev-parse", "--abbrev-ref", "HEAD"), "main")


if __name__ == "__main__":
    unittest.main()
