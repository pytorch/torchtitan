# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from dataclasses import replace

import pytest
import torch
from torch.distributed.tensor import Shard

from torchtitan.components.optim import AdamW, DistMuon, OptimizersContainer
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.flex_shard import BlockShard, Owned
from torchtitan.models.deepseek_v4 import DeepSeekV4Model
from torchtitan_recipes.tests.models.deepseek_v4 import (
    _dist_muon_optimizer,
    deepseek_v4_debugmodel,
    deepseek_v4_mtp_debugmodel,
)


def test_deepseek_v4_uses_dist_muon_with_complete_compute_layouts() -> None:
    config = deepseek_v4_debugmodel()
    assert isinstance(config.model, DeepSeekV4Model.Config)
    muon_config, adamw_config = config.optim.optimizer.optimizers

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
    assert {
        f"layers.2.attention.indexer.{projection}.weight"
        for projection in ("wq_b", "weights_proj", "compressor.wkv", "compressor.wgate")
    } <= muon_fqns
    assert all(not fqn.endswith(".ape") for fqn in muon_fqns)
    compute_shardings = muon_config.compute_sharding_by_fqn
    assert muon_fqns == compute_shardings.keys()
    assert all(
        layout.compute_dtype is torch.bfloat16 for layout in compute_shardings.values()
    )

    owned = compute_shardings["layers.0.attention.wq_a.weight"]
    assert isinstance(owned.shardings_by_mesh_axis["dp_shard"], Owned)
    attention = config.model.layers[0].attention
    nope_dim = attention.head_dim - attention.rope_head_dim
    for projection in ("wq_b", "wkv"):
        layout = compute_shardings[f"layers.0.attention.{projection}.weight"]
        assert layout.shardings_by_mesh_axis["dp_shard"] == BlockShard(
            dim=0, block_sizes=(nope_dim, attention.rope_head_dim)
        )
    assert compute_shardings["layers.0.attention.wo_a.weight"].shardings_by_mesh_axis[
        "dp_shard"
    ] == BlockShard(dim=0, block_sizes=(attention.o_lora_rank,))
    hc_mult = config.model.layers[0].hc_attn_pre.hc_mult
    for module in ("hc_attn_pre", "hc_ffn_pre"):
        assert compute_shardings[f"layers.0.{module}.hc_fn"].shardings_by_mesh_axis[
            "dp_shard"
        ] == BlockShard(dim=0, block_sizes=(hc_mult, hc_mult, hc_mult**2))
    assert isinstance(
        compute_shardings["hc_head.hc_fn"].shardings_by_mesh_axis["dp_shard"],
        Owned,
    )
    shared_w13 = compute_shardings["layers.0.feed_forward.shared_experts.w13.weight"]
    assert shared_w13.shardings_by_mesh_axis["dp_shard"] == Shard(0)
    per_expert = compute_shardings["layers.0.feed_forward.routed_experts.w13.weight"]
    assert per_expert.shardings_by_mesh_axis == {
        "dp_shard": Shard(0),
        "dp_shard_cp": Shard(0),
        "ep": Shard(0),
        "edp_shard": Shard(0),
    }
    assert per_expert.shard_order_by_tensor_dim == {0: ("ep", "edp_shard")}
    assert per_expert.compute_dtype is torch.bfloat16
    csa = config.model.layers[2].attention
    csa_nope = csa.head_dim - csa.rope_head_dim
    assert compute_shardings[
        "layers.2.attention.compressor.wkv.weight"
    ].shardings_by_mesh_axis["dp_shard"] == BlockShard(
        dim=0,
        block_sizes=(csa_nope, csa.rope_head_dim, csa_nope, csa.rope_head_dim),
    )
    assert compute_shardings[
        "layers.2.attention.compressor.wgate.weight"
    ].shardings_by_mesh_axis["dp_shard"] == BlockShard(
        dim=0, block_sizes=(csa.head_dim,)
    )
    hca = config.model.layers[3].attention
    assert compute_shardings[
        "layers.3.attention.compressor_128.wkv.weight"
    ].shardings_by_mesh_axis["dp_shard"] == BlockShard(
        dim=0, block_sizes=(hca.head_dim - hca.rope_head_dim, hca.rope_head_dim)
    )
    assert isinstance(
        compute_shardings[
            "layers.3.attention.compressor_128.wgate.weight"
        ].shardings_by_mesh_axis["dp_shard"],
        Owned,
    )
    indexer = csa.indexer
    assert indexer is not None
    indexer_nope_dim = indexer.index_head_dim - indexer.rope_head_dim
    for projection, block_sizes in (
        ("wq_b", (indexer_nope_dim, indexer.rope_head_dim)),
        (
            "compressor.wkv",
            (
                indexer_nope_dim,
                indexer.rope_head_dim,
                indexer_nope_dim,
                indexer.rope_head_dim,
            ),
        ),
        ("compressor.wgate", (indexer.index_head_dim,)),
    ):
        assert compute_shardings[
            f"layers.2.attention.indexer.{projection}.weight"
        ].shardings_by_mesh_axis["dp_shard"] == BlockShard(
            dim=0, block_sizes=block_sizes
        )
    assert isinstance(
        compute_shardings[
            "layers.2.attention.indexer.weights_proj.weight"
        ].shardings_by_mesh_axis["dp_shard"],
        Owned,
    )


@pytest.mark.parametrize("num_layers", (1, 2, 3, 4))
def test_deepseek_v4_pairs_buckets_after_first_layer(num_layers: int) -> None:
    config = deepseek_v4_debugmodel()
    assert isinstance(config.model, DeepSeekV4Model.Config)
    source_muon = config.optim.optimizer.optimizers[0]
    assert isinstance(source_muon, DistMuon.Config)
    model_config = replace(
        config.model,
        n_layers=num_layers,
        layers=[config.model.layers[index % 4] for index in range(num_layers)],
        compress_ratios=tuple(
            config.model.compress_ratios[index % 4] for index in range(num_layers)
        ),
    )
    muon_config = _dist_muon_optimizer(
        model_config,
        muon_lr=8e-4,
        adamw_lr=8e-4,
        muon_ns_steps=10,
        muon_ns_coefficients=source_muon.ns_coefficients,
    ).optimizers[0]
    assert isinstance(muon_config, DistMuon.Config)
    layer_groups = [(0,)] + [
        tuple(range(first, min(first + 2, num_layers)))
        for first in range(1, num_layers, 2)
    ]
    expected_names = [
        "layers." + "-".join(map(str, group)) + suffix
        for group in layer_groups
        for suffix in ("", ".routed-experts")
    ] + ["hc_head"]
    assert [bucket.name for bucket in muon_config.bucket_configs] == expected_names
    for group, dense, routed in zip(
        layer_groups,
        muon_config.bucket_configs[:-1:2],
        muon_config.bucket_configs[1:-1:2],
        strict=True,
    ):
        prefixes = tuple(f"layers.{layer}." for layer in group)
        expected = {
            fqn
            for fqn in muon_config.compute_sharding_by_fqn
            if fqn.startswith(prefixes)
        }
        assert set(dense.patterns) == {
            fqn for fqn in expected if ".routed_experts." not in fqn
        }
        assert set(routed.patterns) == {
            fqn for fqn in expected if ".routed_experts." in fqn
        }
    patterns = [
        pattern for bucket in muon_config.bucket_configs for pattern in bucket.patterns
    ]
    assert len(patterns) == len(set(patterns))
    assert set(patterns) == set(muon_config.compute_sharding_by_fqn)


def test_deepseek_v4_mtp_parameters_have_dist_muon_buckets() -> None:
    config = deepseek_v4_mtp_debugmodel()
    muon_config = config.optim.optimizer.optimizers[0]
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
    muon_config = config.optim.optimizer.optimizers[0]

    assert isinstance(muon_config, DistMuon.Config)
    assert muon_config.ns_steps == 2
    assert muon_config.ns_coefficients == coefficients


def test_deepseek_v4_tp_pp_guards_apply_only_to_dist_muon() -> None:
    for degree_field in ("tensor_parallel_degree", "pipeline_parallel_degree"):
        config = deepseek_v4_debugmodel(
            parallelism=ParallelismConfig(expert_parallel_degree=2)
        )
        config.optim.optimizer = OptimizersContainer.Config(
            optimizers=[AdamW.Config(pattern=r".*", fused=False)]
        )
        setattr(config.parallelism, degree_field, 2)
        replace(config)
        config.optim.optimizer = deepseek_v4_debugmodel().optim.optimizer
        try:
            replace(config)
        except ValueError as error:
            assert f"{degree_field}=1" in str(error)
        else:
            raise AssertionError(f"DistMuon with {degree_field}=2 must be rejected")
