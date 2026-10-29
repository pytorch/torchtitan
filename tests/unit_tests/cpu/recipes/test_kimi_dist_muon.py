# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from dataclasses import replace

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Replicate, Shard
from torch.distributed.tensor.placement_types import _StridedShard
from torch.testing._internal.distributed.fake_pg import FakeProcessGroup, FakeStore

from torchtitan.components.optim.optimizer import DistMuon
from torchtitan.config.transform import ContextParallelTransform
from torchtitan.distributed.flex_shard import BlockShard
from torchtitan.distributed.fsdp import linear_param_shard_placements
from torchtitan.models.base.module import Module
from torchtitan.models.base.sharding import resolve_placements
from torchtitan.models.common.attention.cp_kda import ContextParallelInnerKDA
from torchtitan.models.common.attention.cp_mla import UlyssesMLAAttention
from torchtitan.models.common.attention.kda import InnerKDA
from torchtitan.models.common.attention.mla import MLAAttention
from torchtitan_recipes.tests.models.kimi_k2_7 import kimi_k2_5_debugmodel
from torchtitan_recipes.tests.models.kimi_k3 import kimi_k3_debugmodel


@pytest.mark.parametrize(
    ("factory", "cp_degree", "tp_degree", "ep_degree"),
    [
        (kimi_k2_5_debugmodel, 1, 2, 4),
        (kimi_k3_debugmodel, 1, 2, 4),
        (kimi_k3_debugmodel, 2, 2, 4),
        (kimi_k2_5_debugmodel, 1, 1, 1),
        (kimi_k3_debugmodel, 2, 1, 1),
    ],
)
def test_kimi_dist_muon_tp_variant_builds_optimizer_without_mutating_base(
    factory, cp_degree, tp_degree, ep_degree
):
    """Resolve actual model parameters through native TP/FSDP policies and DistMuon."""
    original = factory(seq_len=256)
    original = replace(
        original,
        optim=replace(
            original.optim,
            optimizer=replace(
                original.optim.optimizer,
                optimizers=[
                    replace(item, ns_steps=3)
                    if isinstance(item, DistMuon.Config)
                    else item
                    for item in original.optim.optimizer.optimizers
                ],
            ),
        ),
    )
    original_optim = original.optim
    original_optimizer_configs = original_optim.optimizer
    original_muon = next(
        item
        for item in original_optimizer_configs.optimizers
        if isinstance(item, DistMuon.Config)
    )
    original_layouts = deepcopy(original_muon.compute_sharding_by_fqn)
    original_buckets = deepcopy(original_muon.bucket_configs)
    original_pattern = original_muon.pattern
    parallelism = replace(
        original.parallelism,
        data_parallel_shard_degree=2,
        tensor_parallel_degree=tp_degree,
        context_parallel_degree=cp_degree,
        expert_parallel_degree=ep_degree,
        context_parallel_load_balancer=None,
    )
    model_config = factory(
        seq_len=256, parallelism=replace(parallelism, context_parallel_degree=1)
    ).model
    model_config.vision_encoder = None
    if cp_degree > 1:
        ContextParallelTransform(
            inner_attention_map={
                MLAAttention: UlyssesMLAAttention,
                InnerKDA: ContextParallelInnerKDA,
            }
        ).transform(model_config)
    config = replace(original, model=model_config, parallelism=parallelism)
    muon_config = next(
        item
        for item in config.optim.optimizer.optimizers
        if isinstance(item, DistMuon.Config)
    )
    options = FakeProcessGroup.Options()
    options.simulate_uniform_ranks = True
    dist.init_process_group(
        "fake",
        store=FakeStore(),
        rank=0,
        world_size=2 * cp_degree * tp_degree,
        pg_options=options,
    )
    try:
        mesh = init_device_mesh(
            "cpu", (2, cp_degree, tp_degree), mesh_dim_names=("dp_shard", "cp", "tp")
        )
        if cp_degree > 1:
            mesh["dp_shard", "cp"]._flatten("dp_shard_cp")
        dense_mesh = mesh["dp_shard_cp" if cp_degree > 1 else "dp_shard", "tp"]
        if ep_degree > 1:
            expert_mesh = init_device_mesh(
                "cpu",
                (dist.get_world_size() // ep_degree, ep_degree),
                mesh_dim_names=("edp_shard", "ep"),
            )
        with torch.device("meta"):
            model = config.model.build()
        module_configs = {
            fqn: module_config
            for fqn, module_config, _, _ in config.model.traverse(
                Module.Config, recurse=True
            )
        }
        fsdp_placements = linear_param_shard_placements(model)
        parameters = {}
        for fqn, parameter in model.named_parameters():
            if fqn not in muon_config.compute_sharding_by_fqn:
                continue
            if ".routed_experts." in fqn and ep_degree > 1:
                placements = (
                    _StridedShard(0, split_factor=ep_degree)
                    if cp_degree > 1
                    else Replicate(),
                    Shard(0),
                )
                storage_mesh = expert_mesh
            else:
                module_fqn, parameter_name = fqn.rsplit(".", 1)
                sharding_config = module_configs[module_fqn].sharding_config
                tp_placement = resolve_placements(
                    sharding_config.state_shardings[parameter_name], mesh
                )[-1]
                fsdp_placement = fsdp_placements.get(parameter, Shard(0))
                if (
                    isinstance(tp_placement, Shard)
                    and tp_placement.dim == fsdp_placement.dim
                ):
                    fsdp_placement = _StridedShard(
                        fsdp_placement.dim, split_factor=tp_degree
                    )
                placements = (fsdp_placement, tp_placement)
                storage_mesh = dense_mesh
            parameters[fqn] = torch.nn.Parameter(
                distribute_tensor(
                    torch.zeros(parameter.shape), storage_mesh, placements
                )
            )
        assert parameters.keys() == muon_config.compute_sharding_by_fqn.keys()
        optimizer = replace(
            muon_config, ns_device=None, max_nccl_ctas_for_comms=-1
        ).build(
            params=[
                {"params": list(parameters.values()), "param_names": list(parameters)}
            ]
        )
        assert len(optimizer.param_groups[0]["params"]) == len(parameters)
        assert muon_config.ns_steps == original_muon.ns_steps == 3
        dense_dp_axis = "dp_shard_cp" if cp_degree > 1 else "dp_shard"
        for layout in optimizer._parameter_compute_layouts:
            if type(layout.compute_sharding) is BlockShard:
                axes = tuple(
                    layout.param.device_mesh.mesh_dim_names[axis]
                    for axis in layout.redistribution_storage_mesh_axes
                )
                assert axes in ((), (dense_dp_axis,))
            if ".routed_experts." in layout.fqn:
                declared_axes = muon_config.compute_sharding_by_fqn[
                    layout.fqn
                ].shardings_by_mesh_axis
                assert ("tp" in declared_axes) == (ep_degree == 1 and tp_degree > 1)
        assert original.optim is original_optim
        assert original.optim.optimizer is original_optimizer_configs
        assert original_muon.compute_sharding_by_fqn == original_layouts
        assert original_muon.bucket_configs == original_buckets
        assert original_muon.pattern == original_pattern
    finally:
        dist.destroy_process_group()
