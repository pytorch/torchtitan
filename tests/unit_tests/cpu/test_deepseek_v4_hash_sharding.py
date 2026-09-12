# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace

import torch

from torchtitan.config import ParallelismConfig
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import (
    _per_axis_types,
    spmd_validate_redistributions,
)
from torchtitan.models.common.decoder_sharding import (
    token_id_placement,
    token_id_sequence_parallel_placement,
)
from torchtitan.models.common.moe import MoE
from torchtitan.models.deepseek_v4 import model_registry


def _runtime(*, tp: int, ep: int, sp: bool) -> SimpleNamespace:
    return SimpleNamespace(
        parallelism=ParallelismConfig(
            tensor_parallel_degree=tp,
            expert_parallel_degree=ep,
            enable_sequence_parallel=sp,
            context_parallel_degree=1,
            pipeline_parallel_degree=1,
        )
    )


class TestDeepSeekV4HashRoutingSharding(unittest.TestCase):
    def test_uses_common_moe(self):
        model_config = model_registry("debugmodel").model
        moe_config = model_config.layers[0].moe
        assert moe_config is not None
        self.assertIs(type(moe_config), MoE.Config)

        with torch.device("meta"):
            moe = moe_config.build()
        self.assertIs(type(moe), MoE)

    def test_sp_ep_shards_hash_input_ids_with_activations(self):
        model_config = model_registry("debugmodel").model
        model_config.update_from_config(config=_runtime(tp=2, ep=2, sp=True))

        later_moe = model_config.layers[2].moe
        assert later_moe is not None
        later_src = later_moe.sharding_config.in_src_shardings
        later_dst = later_moe.sharding_config.in_dst_shardings
        self.assertNotIn("input_ids_T", later_src)
        self.assertNotIn("input_ids_T", later_dst)

        expected_src = _per_axis_types(token_id_placement())
        expected_dst = _per_axis_types(token_id_sequence_parallel_placement())
        for layer_id in (0, 1):
            hash_moe = model_config.layers[layer_id].moe
            assert hash_moe is not None
            hash_src = hash_moe.sharding_config.in_src_shardings["input_ids_T"]
            hash_dst = hash_moe.sharding_config.in_dst_shardings["input_ids_T"]
            x_dst = hash_moe.sharding_config.in_dst_shardings["x_TD"]
            self.assertEqual(_per_axis_types(hash_src), expected_src)
            self.assertEqual(_per_axis_types(hash_dst), expected_dst)
            self.assertEqual(
                hash_dst.partition_spec,
                token_id_sequence_parallel_placement().partition_spec,
            )
            # Hash ids must take the same TP token shard as MoE activations.
            self.assertEqual(
                _per_axis_types(hash_dst)[MeshAxisName.TP],
                _per_axis_types(x_dst)[MeshAxisName.TP],
            )
            spmd_validate_redistributions(hash_moe.sharding_config)

    def test_sp_without_ep_keeps_hash_input_ids_replicated_on_tp(self):
        model_config = model_registry("debugmodel").model
        model_config.update_from_config(config=_runtime(tp=2, ep=1, sp=True))

        hash_moe = model_config.layers[0].moe
        assert hash_moe is not None
        hash_src = hash_moe.sharding_config.in_src_shardings["input_ids_T"]
        hash_dst = hash_moe.sharding_config.in_dst_shardings["input_ids_T"]
        expected = _per_axis_types(token_id_placement())
        self.assertEqual(_per_axis_types(hash_src), expected)
        self.assertEqual(_per_axis_types(hash_dst), expected)
        spmd_validate_redistributions(hash_moe.sharding_config)

    def test_ep_without_sp_keeps_hash_input_ids_replicated_on_tp(self):
        model_config = model_registry("debugmodel").model
        model_config.update_from_config(config=_runtime(tp=2, ep=2, sp=False))

        hash_moe = model_config.layers[0].moe
        assert hash_moe is not None
        hash_src = hash_moe.sharding_config.in_src_shardings["input_ids_T"]
        hash_dst = hash_moe.sharding_config.in_dst_shardings["input_ids_T"]
        expected = _per_axis_types(token_id_placement())
        self.assertEqual(_per_axis_types(hash_src), expected)
        self.assertEqual(_per_axis_types(hash_dst), expected)
        spmd_validate_redistributions(hash_moe.sharding_config)


if __name__ == "__main__":
    unittest.main()
