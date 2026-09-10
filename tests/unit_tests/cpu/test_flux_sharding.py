# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace

import spmd_types as spmd

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.flux.sharding import set_flux_inner_attention_local_spmd


class TestFluxSharding(unittest.TestCase):
    def test_context_parallel_attention_gathers_keys_and_values(self):
        config = SimpleNamespace(sharding_config=None)
        set_flux_inner_attention_local_spmd(config)

        sharding = config.sharding_config
        assert sharding is not None
        assert sharding.in_src_shardings is not None
        assert sharding.in_dst_shardings is not None
        for name in ("k_BLHK", "v_BLHV"):
            src = sharding.in_src_shardings[name].local_type[MeshAxisName.CP]
            dst = sharding.in_dst_shardings[name].local_type[MeshAxisName.CP]
            self.assertEqual(src, spmd.S(1))
            self.assertEqual(dst, spmd.R)


if __name__ == "__main__":
    unittest.main()
