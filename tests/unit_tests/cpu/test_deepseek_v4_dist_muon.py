# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from dataclasses import replace

import torch
from torch.distributed.tensor import Shard

from torchtitan.components.optimizer import AdamW, DistMuon
from torchtitan.distributed.flex_shard import BlockShard, Owned
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.deepseek_v4.config_registry import (
    deepseek_v4_debugmodel,
    deepseek_v4_mtp_debugmodel,
)


def test_deepseek_v4_uses_dist_muon_with_complete_compute_layouts() -> None:
    config = deepseek_v4_debugmodel()
    muon_config, adamw_config = config.optimizer.optimizers

    assert isinstance(muon_config, DistMuon.Config)
    assert isinstance(adamw_config, AdamW.Config)
    assert muon_config.ns_steps == 10
    assert muon_config.ns_coefficients == (
        ((3.4445, -4.7750, 2.0315),) * 8 + ((2.0, -1.5, 0.5),) * 2
    )

    with torch.device("meta"):
        model = config.model.build()
    muon_fqns = {
        fqn
        for fqn, parameter in model.named_parameters()
        if parameter.requires_grad and re.search(muon_config.pattern, fqn)
    }
    assert all(".attention.indexer." not in fqn for fqn in muon_fqns)
    compute_shardings = muon_config.compute_sharding_by_fqn
    assert muon_fqns == compute_shardings.keys()

    owned = compute_shardings["layers.0.attention.wq_a.weight"]
    assert isinstance(owned.shardings_by_mesh_axis["dp_shard"], Owned)
    per_head = compute_shardings["layers.0.attention.wq_b.weight"]
    assert per_head.shardings_by_mesh_axis["dp_shard"] == BlockShard(
        dim=0,
        block_sizes=(config.model.layers[0].attention.head_dim,),
    )
    shared_w13 = compute_shardings["layers.0.moe.shared_experts.w13.weight"]
    assert shared_w13.shardings_by_mesh_axis["dp_shard"] == Shard(0)
    per_expert = compute_shardings["layers.0.moe.routed_experts.w13.weight"]
    assert per_expert.shardings_by_mesh_axis["dp_shard"] == Shard(0)

    buckets = muon_config.bucket_configs
    bucket_patterns = [pattern for bucket in buckets for pattern in bucket.patterns]
    assert set(bucket_patterns) == set(compute_shardings)
    assert len(bucket_patterns) == len(set(bucket_patterns))
    for bucket in buckets:
        routed = [".moe.routed_experts." in pattern for pattern in bucket.patterns]
        assert all(routed) or not any(routed)


def test_deepseek_v4_mtp_parameters_have_dist_muon_buckets() -> None:
    config = deepseek_v4_mtp_debugmodel()
    muon_config = config.optimizer.optimizers[0]
    assert isinstance(muon_config, DistMuon.Config)
    compute_shardings = muon_config.compute_sharding_by_fqn
    bucket_names = {bucket.name for bucket in muon_config.bucket_configs}

    assert "mtp_layers.0.e_proj.weight" in compute_shardings
    assert "mtp_layers.0.hc_head.hc_fn" in compute_shardings
    assert "mtp_layers.0" in bucket_names
    assert "mtp_layers.0.routed-experts" in bucket_names
    assert "hc_head" in bucket_names


def test_deepseek_v4_recipe_configures_per_step_ns_coefficients() -> None:
    coefficients = (
        (3.4445, -4.7750, 2.0315),
        (3.0, -4.0, 2.0),
    )
    config = deepseek_v4_debugmodel(
        muon_ns_steps=2,
        muon_ns_coefficients=coefficients,
    )
    muon_config = config.optimizer.optimizers[0]

    assert isinstance(muon_config, DistMuon.Config)
    assert muon_config.ns_steps == 2
    assert muon_config.ns_coefficients == coefficients


def test_deepseek_v4_aligns_expert_layout_after_parallelism_change() -> None:
    config = deepseek_v4_debugmodel()
    config.parallelism.expert_parallel_degree = 2
    config = replace(config)
    muon_config = config.optimizer.optimizers[0]
    assert isinstance(muon_config, DistMuon.Config)
    compute_shardings = muon_config.compute_sharding_by_fqn
    per_expert = compute_shardings["layers.0.moe.routed_experts.w13.weight"]

    assert per_expert.shardings_by_mesh_axis == {
        MeshAxisName.EDP_SHARD.value: Shard(0),
        MeshAxisName.EP.value: Shard(0),
    }
    assert per_expert.shard_order_by_tensor_dim == {
        0: (MeshAxisName.EP.value, MeshAxisName.EDP_SHARD.value),
    }
