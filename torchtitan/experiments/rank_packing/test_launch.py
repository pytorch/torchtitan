# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU-only regression tests for rank-launch cleanup."""

import subprocess
import unittest
from unittest.mock import patch

from .launch import _cleanup


class LaunchCleanupTest(unittest.TestCase):
    def test_exit_between_probe_and_signal(self):
        process = unittest.mock.Mock(pid=123)
        with (
            patch(
                "torchtitan.experiments.rank_packing.launch._group_exists",
                return_value=True,
            ),
            patch(
                "torchtitan.experiments.rank_packing.launch.os.killpg",
                side_effect=ProcessLookupError,
            ),
        ):
            _cleanup(process)

    def test_already_exited_group(self):
        process = unittest.mock.Mock(pid=123)
        with (
            patch(
                "torchtitan.experiments.rank_packing.launch._group_exists",
                return_value=False,
            ),
            patch("torchtitan.experiments.rank_packing.launch.os.killpg") as kill,
        ):
            _cleanup(process)
            kill.assert_not_called()

    def test_cleanup_of_cpu_child(self):
        process = subprocess.Popen(["sleep", "60"], start_new_session=True)
        _cleanup(process)
        self.assertIsNotNone(process.poll())


if __name__ == "__main__":
    unittest.main()
