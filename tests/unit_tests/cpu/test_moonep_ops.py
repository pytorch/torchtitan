# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.distributed.moonep import ops


class TestMoonEPPlanTable(unittest.TestCase):
    def test_a_dispatch_after_one_left_uncombined_is_refused(self):
        ops._plans[0] = ops._Plan(plan=None)
        try:
            with self.assertRaisesRegex(RuntimeError, r"dispatches \[0\] were never"):
                ops._dispatch(
                    torch.zeros(1, 4, dtype=torch.bfloat16),
                    torch.zeros(1, 1),
                    torch.zeros(1, 1, dtype=torch.int32),
                    torch.zeros(1, dtype=torch.int32),
                )
        finally:
            ops._plans.clear()


if __name__ == "__main__":
    unittest.main()
