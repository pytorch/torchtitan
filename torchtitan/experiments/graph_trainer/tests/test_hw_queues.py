# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchtitan.experiments.graph_trainer.hw_queues import _next_pow2, _stream_lanes


class TestHwQueues(unittest.TestCase):
    def test_ep2_dsv3_with_fsdp_overlap(self):
        lanes = _stream_lanes(
            dp_shard_active=True,
            is_moe=True,
            ep=2,
            tp=1,
            cp=1,
            fsdp_ag_rs_overlap=True,
            cudagraph=True,
        )
        self.assertNotIn("cudagraph_capture", lanes)
        self.assertEqual(len(lanes), 5)
        self.assertEqual(_next_pow2(len(lanes)), 8)

    def test_ep2_dsv3_with_cudagraph(self):
        lanes = _stream_lanes(
            dp_shard_active=True,
            is_moe=True,
            ep=2,
            tp=1,
            cp=1,
            fsdp_ag_rs_overlap=False,
            cudagraph=True,
        )
        self.assertIn("cudagraph_capture", lanes)
        self.assertEqual(len(lanes), 5)
        self.assertEqual(_next_pow2(len(lanes)), 8)

    def test_plain_fsdp(self):
        lanes = _stream_lanes(
            dp_shard_active=True,
            is_moe=False,
            ep=1,
            tp=1,
            cp=1,
            fsdp_ag_rs_overlap=False,
            cudagraph=False,
        )
        self.assertEqual(lanes, ["compute", "all_reduce", "fsdp_comm"])
        self.assertEqual(_next_pow2(len(lanes)), 4)

    def test_next_pow2(self):
        self.assertEqual((_next_pow2(7), _next_pow2(8), _next_pow2(9)), (8, 8, 16))


if __name__ == "__main__":
    unittest.main()
