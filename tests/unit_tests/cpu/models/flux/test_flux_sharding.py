# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import spmd_types as spmd

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.flux.sharding import flux_inner_attention_config


class TestFluxSharding(unittest.TestCase):
    def test_context_parallel_attention_gathers_keys_and_values(self):
        sharding = flux_inner_attention_config()
        assert sharding.in_src_shardings is not None
        assert sharding.in_dst_shardings is not None
        for name in ("k_BLHK", "v_BLHV"):
            src = sharding.in_src_shardings[name].local_type[MeshAxisName.CP]
            dst = sharding.in_dst_shardings[name].local_type[MeshAxisName.CP]
            self.assertEqual(src, spmd.S(1))
            self.assertEqual(dst, spmd.R)


if __name__ == "__main__":
    unittest.main()
