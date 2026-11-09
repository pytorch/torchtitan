# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.models.common.moe.token_dispatcher import LocalTokenDispatcher


class TestArgsortExpertIds(unittest.TestCase):
    def test_narrow_key_argsort_matches_int64_argsort(self):
        # Each side of the uint8 and int16 thresholds, with the largest id present:
        # an off-by-one in a threshold would wrap that id and change the order.
        torch.manual_seed(0)
        for num_experts in (256, 257, 2**15, 2**15 + 1):
            dispatcher = LocalTokenDispatcher.Config(
                num_experts=num_experts, top_k=1
            ).build()
            ids_TK = torch.randint(0, num_experts, (4096, 1))
            ids_TK[0] = num_experts - 1
            _, token_indices_experts_sorted_N, _ = dispatcher._local_reorder(
                torch.randn(4096, 2), torch.rand(4096, 1), ids_TK
            )
            self.assertTrue(
                torch.equal(
                    token_indices_experts_sorted_N,
                    torch.argsort(ids_TK.view(-1), stable=True),
                )
            )


if __name__ == "__main__":
    unittest.main()
