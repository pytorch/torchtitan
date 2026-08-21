# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The CP contracts must describe what the CP code actually does.

A declaration that only agrees with itself is worth nothing -- these tests tie
each contract to the implementation it claims to describe, so the two cannot
drift apart silently.
"""

import unittest

import spmd_types as spmd
import torch

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.kimi_k3.sharding import (
    contract_for_mode,
    HEAD_DIM,
    KCP,
    SEQ_DIM,
    ULYSSES,
)


CP = MeshAxisName.CP


class TestCPContracts(unittest.TestCase):
    def test_ulysses_pair_matches_the_all_to_all_reshape(self):
        """S(1) -> S(2) is exactly what _cp_all_to_all_headseq does to the shape.

        Run with a stub for the collective: on a single process the all-to-all is
        the identity, so the surrounding reshape/permute is what gets checked --
        and that is the half the contract describes.
        """
        from torchtitan.models.kimi_k3 import model as k3_model

        cp, B, t_loc, num_heads, K = 4, 2, 8, 12, 6
        x = torch.randn(B, t_loc, num_heads, K)

        class _Stub:
            pass

        real_ws = k3_model.dist.get_world_size
        import torch.distributed.nn.functional as dist_nn

        real_a2a = dist_nn.all_to_all_single
        k3_model.dist.get_world_size = lambda group: cp
        dist_nn.all_to_all_single = lambda out, inp, group=None: inp
        try:
            fwd = k3_model._cp_all_to_all_headseq(
                x, _Stub(), src_dim=SEQ_DIM, dst_dim=HEAD_DIM
            )
            # in_src S(1): sequence is the sharded axis, so t_loc is a shard.
            # in_dst S(2): heads become the sharded axis, sequence goes full.
            self.assertEqual(fwd.shape, (B, cp * t_loc, num_heads // cp, K))
            back = k3_model._cp_all_to_all_headseq(
                fwd, _Stub(), src_dim=HEAD_DIM, dst_dim=SEQ_DIM
            )
            self.assertEqual(back.shape, x.shape)
        finally:
            k3_model.dist.get_world_size = real_ws
            dist_nn.all_to_all_single = real_a2a

        self.assertEqual(ULYSSES.in_src.axis_types[CP], spmd.S(SEQ_DIM))
        self.assertEqual(ULYSSES.in_dst.axis_types[CP], spmd.S(HEAD_DIM))
        # out pair is the same swap reversed, so a round trip lands where it started
        self.assertEqual(ULYSSES.out_dst.axis_types[CP], ULYSSES.in_src.axis_types[CP])

    def test_the_contract_actually_drives_the_all_to_all(self):
        """A contract naming an unimplemented pair must fail, not be ignored.

        This is what makes the declaration load-bearing. Before the dims came from
        the contract, ``_forward_cp`` hard-coded the direction, so editing ULYSSES
        to name any other pair changed precisely nothing at runtime.
        """
        from torchtitan.models.kimi_k3 import model as k3_model

        x = torch.randn(2, 8, 12, 6)
        with self.assertRaises(ValueError) as cm:
            k3_model._cp_all_to_all_headseq(x, object(), src_dim=SEQ_DIM, dst_dim=3)
        self.assertIn("no Ulysses all-to-all", str(cm.exception))

        # And the pair the contract actually names is one of the implemented ones.
        self.assertIn(ULYSSES.in_dims(), ((SEQ_DIM, HEAD_DIM), (HEAD_DIM, SEQ_DIM)))
        self.assertEqual(ULYSSES.out_dims(), tuple(reversed(ULYSSES.in_dims())))

    def test_kcp_is_an_identity_pair(self):
        # KCP keeps the sequence sharded end to end, so the boundary moves no data.
        self.assertFalse(KCP.redistributes())
        self.assertTrue(ULYSSES.redistributes())

    def test_only_ulysses_asks_for_a_head_split(self):
        # The head-divisibility precondition is driven off this flag; if KCP ever
        # reports True the wiring starts rejecting configurations that work.
        self.assertTrue(ULYSSES.head_sharded)
        self.assertFalse(KCP.head_sharded)

    def test_modes_match_the_config_field(self):
        for mode in ("ulysses", "kcp"):
            self.assertEqual(contract_for_mode(mode).name, mode)
        with self.assertRaises(ValueError):
            contract_for_mode("ring")


if __name__ == "__main__":
    unittest.main()
