# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

import torchtitan.models.common.moe.token_dispatcher as token_dispatcher


class TestArgsortExpertIds(unittest.TestCase):
    def test_narrow_key_argsort_matches_int64_argsort(self):
        # Each side of the uint8 and int16 thresholds, with the largest id present:
        # an off-by-one in a threshold would wrap that id and change the order.
        torch.manual_seed(0)
        for num_experts in (256, 257, 2**15, 2**15 + 1):
            ids_N = torch.randint(0, num_experts, (4096,))
            ids_N[0] = num_experts - 1
            self.assertTrue(
                torch.equal(
                    token_dispatcher._argsort_expert_ids(ids_N, num_experts),
                    torch.argsort(ids_N, stable=True),
                )
            )


if __name__ == "__main__":
    unittest.main()
