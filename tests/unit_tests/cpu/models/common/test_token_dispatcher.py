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
        torch.manual_seed(0)
        for num_experts in (5, 300, 40000):  # uint8, int16 and int64 keys
            ids_N = torch.randint(0, num_experts, (4096,))
            self.assertTrue(
                torch.equal(
                    token_dispatcher._argsort_expert_ids(ids_N, num_experts),
                    torch.argsort(ids_N, stable=True),
                )
            )


if __name__ == "__main__":
    unittest.main()
