# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The CP contracts, on CPU.

These pin the folded token layout. This model carries no batch axis, so the
Ulysses pair moves the shard between tensor dims 0 and 1; the batched spelling
of the same contract used dims 1 and 2, and nothing in a shape check would
catch that swap -- both dims exist and the all-to-all would produce a
plausible tensor with the heads and the sequence exchanged.
"""

from __future__ import annotations

import unittest

import torch

from torchtitan.models.kimi_k3.sharding import (
    contract_for_mode,
    cp_all_to_all_headseq,
    HEAD_DIM,
    KCP,
    SEQ_DIM,
    ULYSSES,
)


class TestCPContracts(unittest.TestCase):
    def test_dims_are_the_folded_ones(self):
        self.assertEqual((SEQ_DIM, HEAD_DIM), (0, 1))

    def test_ulysses_swaps_the_sharded_axis(self):
        self.assertEqual(ULYSSES.in_dims(), (SEQ_DIM, HEAD_DIM))
        self.assertEqual(ULYSSES.out_dims(), (HEAD_DIM, SEQ_DIM))
        self.assertTrue(ULYSSES.redistributes())
        self.assertTrue(ULYSSES.head_sharded)

    def test_kcp_is_an_identity_pair(self):
        """The recurrence passes state rank to rank, which is a sequential
        dependency rather than a redistribution, so no placement pair
        describes it and the contract is declared as an identity."""
        self.assertEqual(KCP.in_dims(), (SEQ_DIM, SEQ_DIM))
        self.assertFalse(KCP.redistributes())
        self.assertFalse(KCP.head_sharded)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            contract_for_mode("ring")

    def test_unimplemented_dim_pair_raises(self):
        """A contract naming a pair with no implementation must raise here
        rather than being quietly ignored."""
        x = torch.zeros(4, 2, 3)
        with self.assertRaises(ValueError):
            cp_all_to_all_headseq(x, None, src_dim=SEQ_DIM, dst_dim=2)


if __name__ == "__main__":
    unittest.main()
