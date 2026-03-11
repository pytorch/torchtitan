# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.distributed.varlen_cp.dispatch_ops import (
    compute_local_cu_seqlens,
    shard_sequence,
)


class TestComputeLocalCuSeqlens(unittest.TestCase):
    def test_doc_within_chunk(self):
        """Document entirely within the chunk."""
        global_cu = torch.tensor([0, 128, 256], dtype=torch.int32)
        local_cu, max_seqlen = compute_local_cu_seqlens(global_cu, 0, 128)
        self.assertEqual(local_cu.tolist(), [0, 128])
        self.assertEqual(max_seqlen, 128)

    def test_doc_spanning_chunk(self):
        """Document spans across the chunk boundary."""
        global_cu = torch.tensor([0, 300, 512], dtype=torch.int32)
        # Chunk [0, 256): contains part of doc 0 (0-256)
        local_cu, max_seqlen = compute_local_cu_seqlens(global_cu, 0, 256)
        self.assertEqual(local_cu.tolist(), [0, 256])
        self.assertEqual(max_seqlen, 256)

        # Chunk [256, 512): contains rest of doc 0 (256-300) and all of doc 1 (300-512)
        local_cu, max_seqlen = compute_local_cu_seqlens(global_cu, 256, 512)
        self.assertEqual(local_cu.tolist(), [0, 44, 256])
        self.assertEqual(max_seqlen, 212)  # doc 1 has 212 tokens in this chunk

    def test_multiple_docs_in_chunk(self):
        """Multiple documents fit within one chunk."""
        global_cu = torch.tensor([0, 64, 128, 192, 256], dtype=torch.int32)
        local_cu, max_seqlen = compute_local_cu_seqlens(global_cu, 0, 256)
        self.assertEqual(local_cu.tolist(), [0, 64, 128, 192, 256])
        self.assertEqual(max_seqlen, 64)

    def test_chunk_with_no_doc_boundaries(self):
        """Chunk is entirely within a single document."""
        global_cu = torch.tensor([0, 512], dtype=torch.int32)
        local_cu, max_seqlen = compute_local_cu_seqlens(global_cu, 128, 384)
        self.assertEqual(local_cu.tolist(), [0, 256])
        self.assertEqual(max_seqlen, 256)


class TestShardSequence(unittest.TestCase):
    def test_basic_sharding(self):
        x = torch.arange(8).float()
        shard_0 = shard_sequence(x, cp_rank=0, cp_world_size=2, seq_dim=0)
        shard_1 = shard_sequence(x, cp_rank=1, cp_world_size=2, seq_dim=0)
        torch.testing.assert_close(shard_0, torch.tensor([0.0, 1, 2, 3]))
        torch.testing.assert_close(shard_1, torch.tensor([4.0, 5, 6, 7]))

    def test_2d_sharding(self):
        x = torch.arange(16).float().reshape(2, 8)
        shard_0 = shard_sequence(x, cp_rank=0, cp_world_size=2, seq_dim=1)
        shard_1 = shard_sequence(x, cp_rank=1, cp_world_size=2, seq_dim=1)
        self.assertEqual(shard_0.shape, (2, 4))
        self.assertEqual(shard_1.shape, (2, 4))


if __name__ == "__main__":
    unittest.main()
