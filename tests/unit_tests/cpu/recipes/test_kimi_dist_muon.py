# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from dataclasses import replace

import pytest
from torch.distributed.tensor import Shard

from torchtitan.components.optim.optimizer import DistMuon
from torchtitan.config.transform.context_parallel import ContextParallelTransform
from torchtitan.distributed.flex_shard import BlockShard, Owned
from torchtitan.models.common.attention.cp_kda import UlyssesInnerKDA
from torchtitan.models.common.attention.cp_mla import UlyssesMLAAttention
from torchtitan.models.common.attention.kda import InnerKDA
from torchtitan.models.common.attention.mla import MLAAttention
from torchtitan_recipes.tests.models.kimi_k2_7 import kimi_k2_5_debugmodel
from torchtitan_recipes.tests.models.kimi_k3 import kimi_k3_debugmodel


@pytest.mark.parametrize("factory", [kimi_k2_5_debugmodel, kimi_k3_debugmodel])
@pytest.mark.parametrize(("tp_degree", "cp_degree"), [(2, 1), (4, 1), (2, 2)])
def test_kimi_dist_muon_tp_preserves_projection_layouts(factory, tp_degree, cp_degree):
    """Variant construction preserves projection policies and the original config."""
    original = factory(seq_len=256)
    original_optim = original.optim
    original_optimizer_configs = original_optim.optimizer
    model = deepcopy(original.model)
    if cp_degree > 1:
        ContextParallelTransform(
            inner_attention_map={
                MLAAttention: UlyssesMLAAttention,
                InnerKDA: UlyssesInnerKDA,
            }
        ).transform(model)
    config = replace(
        original,
        model=model,
        parallelism=replace(
            original.parallelism,
            data_parallel_shard_degree=2,
            context_parallel_degree=cp_degree,
            tensor_parallel_degree=tp_degree,
            expert_parallel_degree=4,
        ),
    )
    assert original.optim is original_optim
    assert original_optim.optimizer is original_optimizer_configs
    optimizer = next(
        item
        for item in config.optim.optimizer.optimizers
        if isinstance(item, DistMuon.Config)
    )
    layouts = optimizer.compute_sharding_by_fqn
    storage_axis = "dp_shard_cp" if cp_degree > 1 else "dp_shard"
    for fqn, layout in layouts.items():
        if ".routed_experts." in fqn:
            assert layout.shard_order_by_tensor_dim[0] == ("ep", "edp_shard")
            continue
        tp_sharding = layout.shardings_by_mesh_axis["tp"]
        assert tp_sharding == layout.shardings_by_mesh_axis[storage_axis]
        assert type(tp_sharding) in (BlockShard, Owned, Shard)
        assert "dp_shard" in layout.shardings_by_mesh_axis
        assert "dp_shard_cp" in layout.shardings_by_mesh_axis

    assert (
        type(layouts["layers.0.feed_forward.w13.weight"].shardings_by_mesh_axis["tp"])
        is Shard
    )
    assert (
        type(layouts["layers.0.feed_forward.w2.weight"].shardings_by_mesh_axis["tp"])
        is Owned
    )
    for layer_id, layer in enumerate(original.model.layers):
        if hasattr(layer.attention, "q_lora_rank") and layer.attention.q_lora_rank:
            query_layout = layouts[f"layers.{layer_id}.attention.wq_b.weight"]
            assert query_layout.shardings_by_mesh_axis["tp"].block_sizes == (
                layer.attention.qk_nope_head_dim,
                layer.attention.qk_rope_head_dim,
            )
    original_optimizer = next(
        item
        for item in original_optim.optimizer.optimizers
        if isinstance(item, DistMuon.Config)
    )
    for fqn, layout in original_optimizer.compute_sharding_by_fqn.items():
        if ".routed_experts." not in fqn:
            assert "dp_shard" in layout.shardings_by_mesh_axis
    assert original.parallelism.context_parallel_degree == 1
