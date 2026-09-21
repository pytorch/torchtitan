# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace

import pytest
import torch
from torch.distributed.tensor import Shard

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.distributed.flex_shard import Owned
from torchtitan.models.kimi_k2_7 import config_registry as kimi_configs


@pytest.mark.parametrize(
    "recipe_name",
    ("kimi_k2_5_debugmodel", "moonlight_16b_a3b", "kimi_vl_a3b", "kimi_k2_5"),
)
def test_kimi_dist_muon_assignments_match_compute_layouts(recipe_name):
    config = getattr(kimi_configs, recipe_name)(seq_len=128)
    with torch.device("meta"):
        model = config.model_spec.model.build()

    optimizer_config = config.optimizer
    groups_by_optimizer, _ = OptimizersContainer._build_param_groups(
        model,
        optimizer_config.param_groups,
        OptimizersContainer._build_impl_kwargs(optimizer_config),
    )
    names_by_optimizer = {
        name: {fqn for group in groups for fqn in group["param_names"]}
        for name, groups in groups_by_optimizer.items()
    }
    muon_names = names_by_optimizer["DistMuon"]
    adamw_names = names_by_optimizer["AdamW"]
    named_parameters = dict(model.named_parameters())
    factory_kwargs = optimizer_config.optimizer_factory_kwargs_by_name["DistMuon"]
    compute_layouts = factory_kwargs["compute_sharding_by_fqn"]

    assert muon_names == set(compute_layouts)
    assert muon_names.isdisjoint(adamw_names)
    assert muon_names | adamw_names == set(named_parameters)
    bucket_fqns = [
        fqn for bucket in factory_kwargs["bucket_configs"] for fqn in bucket.patterns
    ]
    assert len(bucket_fqns) == len(set(bucket_fqns))
    assert set(bucket_fqns) == muon_names

    for layer_id in model.layers:
        prefix = (
            f"layers.{layer_id}.feed_forward"
            if layer_id == "0"
            else f"layers.{layer_id}.moe.shared_experts"
        )
        w13_fqn = f"{prefix}.w13.weight"
        assert w13_fqn in adamw_names
        assert w13_fqn not in compute_layouts
        assert named_parameters[w13_fqn].ndim == 3
        assert named_parameters[w13_fqn].shape[0] == 2
        w2_fqn = f"{prefix}.w2.weight"
        assert w2_fqn in muon_names
        assert named_parameters[w2_fqn].ndim == 2
        assert compute_layouts[w2_fqn].shardings_by_mesh_axis == {"dp_shard": Owned()}

    assert {
        "layers.0.attention.wo.weight",
        "layers.1.moe.router.gate.weight",
        "layers.1.moe.routed_experts.inner_experts.w1_EFD",
        "layers.1.moe.routed_experts.inner_experts.w2_EDF",
        "layers.1.moe.routed_experts.inner_experts.w3_EFD",
    } <= muon_names
    assert {
        "tok_embeddings.weight",
        "norm.weight",
        "lm_head.weight",
        "layers.0.attention_norm.weight",
        "layers.0.ffn_norm.weight",
    } <= adamw_names
    vision_names = {
        fqn for fqn in named_parameters if fqn.startswith("vision_encoder.")
    }
    assert vision_names <= adamw_names


def test_kimi_expert_parallel_override_preserves_feed_forward_compute_layouts():
    config = kimi_configs.kimi_k2_5_debugmodel(seq_len=128)
    config = replace(
        config,
        parallelism=replace(config.parallelism, expert_parallel_degree=8),
    )
    compute_layouts = config.optimizer.optimizer_factory_kwargs_by_name["DistMuon"][
        "compute_sharding_by_fqn"
    ]
    for prefix in ("layers.0.feed_forward", "layers.1.moe.shared_experts"):
        assert f"{prefix}.w13.weight" not in compute_layouts
        assert compute_layouts[f"{prefix}.w2.weight"].shardings_by_mesh_axis == {
            "dp_shard": Owned()
        }
    expert_layout = compute_layouts["layers.1.moe.routed_experts.inner_experts.w1_EFD"]
    assert expert_layout.shardings_by_mesh_axis == {"efsdp": Shard(0), "ep": Shard(0)}
    assert expert_layout.shard_order_by_tensor_dim == {0: ("ep", "efsdp")}
