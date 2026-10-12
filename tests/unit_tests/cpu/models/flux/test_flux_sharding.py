# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import spmd_types as spmd
import torch

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.flux.attention import FluxInnerAttention
from torchtitan.models.flux.sharding import flux_inner_attention_sharding_config


class TestFluxSharding(unittest.TestCase):
    def test_context_parallel_attention_gathers_keys_and_values(self):
        sharding = flux_inner_attention_sharding_config()
        assert sharding.in_src_shardings is not None
        assert sharding.in_dst_shardings is not None
        for name in ("k_BLHK", "v_BLHV"):
            src = sharding.in_src_shardings[name].local_type[MeshAxisName.CP]
            dst = sharding.in_dst_shardings[name].local_type[MeshAxisName.CP]
            self.assertEqual(src, spmd.S(1))
            self.assertEqual(dst, spmd.R)

    def test_varlen_boundaries_preserve_batch_layout(self):
        config = FluxInnerAttention.Config()
        self.assertEqual(config.varlen_attention.window_size, (-1, -1))
        attention = config.build()
        q, k, v = [torch.randn(2, length, 4, 8) for length in (3, 8, 8)]
        with patch.object(
            attention.varlen_attention, "forward", return_value=q.flatten(0, 1)
        ) as kernel:
            actual = attention(q, k, v)

        for packed, batched in zip(kernel.call_args.args, (q, k, v)):
            torch.testing.assert_close(packed, batched.flatten(0, 1))
        metadata = kernel.call_args.kwargs["attention_metadata"]
        torch.testing.assert_close(
            metadata.cu_seq_q, torch.tensor([0, 3, 6], dtype=torch.int32)
        )
        torch.testing.assert_close(
            metadata.cu_seq_k, torch.tensor([0, 8, 16], dtype=torch.int32)
        )
        self.assertEqual((metadata.max_q, metadata.max_k), (3, 8))
        torch.testing.assert_close(actual, q)


if __name__ == "__main__":
    unittest.main()
