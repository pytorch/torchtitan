# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Minimal check for the --rtol relative-tolerance branch in loss_compare."""

import unittest

from scripts.loss_compare import assert_losses_equal


class TestLossCompareRtol(unittest.TestCase):
    def test_exact_default_rejects_tiny_drift(self):
        with self.assertRaises(SystemExit):
            assert_losses_equal({1: 8.048322677612305}, {1: 8.04832935333252})

    def test_rtol_accepts_tiny_drift(self):
        # ~8.3e-7 relative drift, well within 1e-5
        assert_losses_equal({1: 8.048322677612305}, {1: 8.04832935333252}, rtol=1e-5)

    def test_rtol_still_rejects_large_drift(self):
        with self.assertRaises(SystemExit):
            assert_losses_equal({1: 8.0}, {1: 8.01}, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
