# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch.distributed.tensor.experimental._attention import (
    _HeadTailLoadBalancer,
    _PerDocumentHeadTailLoadBalancer,
)


class TestTokenMajorHeadTailLoadBalancing(unittest.TestCase):
    def test_preserves_per_sequence_cp_shards(self):
        num_sequences = 3
        seq_len = 16
        cp_degree = 2
        tokens_BL = torch.arange(num_sequences * seq_len).view(num_sequences, seq_len)

        row_balancer = _HeadTailLoadBalancer(seq_len, cp_degree, "cpu")
        row_indices = row_balancer._generate_indices().expand(num_sequences, -1)
        reordered_BL = tokens_BL.gather(1, row_indices.to(torch.int64))

        flat_balancer = _PerDocumentHeadTailLoadBalancer(
            [[seq_len] * num_sequences], cp_degree, "cpu"
        )
        flat_indices = flat_balancer._generate_indices()[0]
        reordered_T = tokens_BL.flatten()[flat_indices]

        for cp_rank in range(cp_degree):
            expected = reordered_BL.chunk(cp_degree, dim=1)[cp_rank].flatten()
            actual = reordered_T.chunk(cp_degree)[cp_rank]
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
