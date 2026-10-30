# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared DistMuon expert-layout policies for Kimi recipes."""

from dataclasses import replace
from typing import cast

import torch
from torch.distributed.tensor import Shard

from torchtitan.components.optim import DistMuon, OptimizersContainer
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.flex_shard import ComputeLayout
from torchtitan.distributed.parallelism_context import MeshAxisName


def per_expert_compute_layout(parallelism: ParallelismConfig) -> ComputeLayout:
    """Build BF16 per-expert compute ownership from the sparse mesh axes."""
    ep_size = parallelism.expert_parallel_degree
    if ep_size <= 0:
        raise ValueError("expert_parallel_degree must be positive")
    if ep_size == 1:
        # FSDP flattens dp_shard and cp into one storage axis under CP.
        return ComputeLayout(
            shardings_by_mesh_axis={
                MeshAxisName.DP_SHARD.value: Shard(0),
                f"{MeshAxisName.DP_SHARD.value}_{MeshAxisName.CP.value}": Shard(0),
                MeshAxisName.TP.value: Shard(0),
            },
            compute_dtype=torch.bfloat16,
        )

    # Preserve exact EP-first DTensor ownership. If an EP-local expert count is
    # smaller than the edp_shard size, add balanced rank assignment only after
    # benchmarks show that the fixed nonempty edp_shard coordinates are a hotspot.
    return ComputeLayout(
        shardings_by_mesh_axis={
            MeshAxisName.EDP_SHARD.value: Shard(0),
            MeshAxisName.EP.value: Shard(0),
        },
        # EP splits the expert dimension first, then edp_shard repartitions each
        # EP-local expert domain, which reverses the storage-mesh axis order.
        shard_order_by_tensor_dim={
            0: (MeshAxisName.EP.value, MeshAxisName.EDP_SHARD.value),
        },
        compute_dtype=torch.bfloat16,
    )


def align_dist_muon_expert_compute_layouts(
    optimizer_config: OptimizersContainer.Config,
    *,
    parallelism: ParallelismConfig,
) -> OptimizersContainer.Config:
    """Align routed-expert layouts with the final parallelism config.

    Config variants can replace the base recipe's parallelism before invoking
    ``__post_init__``. That parallelism decides whether routed experts use the
    1D ``dp_shard`` layout or the 2D ep/edp_shard layout, so their layouts have
    to be rebuilt here.
    """
    dist_muon_index = next(
        (
            index
            for index, config in enumerate(optimizer_config.optimizers)
            if isinstance(config, DistMuon.Config)
        ),
        None,
    )
    if dist_muon_index is None:
        return optimizer_config
    dist_muon_config = cast(
        DistMuon.Config,
        optimizer_config.optimizers[dist_muon_index],
    )
    compute_sharding_by_fqn = cast(
        dict[str, ComputeLayout],
        dist_muon_config.compute_sharding_by_fqn,
    )
    per_expert = per_expert_compute_layout(parallelism)
    aligned_shardings = {}
    changed = False
    for fqn, compute_layout in compute_sharding_by_fqn.items():
        if ".routed_experts." in fqn and compute_layout != per_expert:
            aligned_shardings[fqn] = per_expert
            changed = True
        else:
            aligned_shardings[fqn] = compute_layout
    if not changed:
        return optimizer_config

    optimizers = list(optimizer_config.optimizers)
    optimizers[dist_muon_index] = replace(
        dist_muon_config,
        compute_sharding_by_fqn=aligned_shardings,
    )
    return replace(
        optimizer_config,
        optimizers=optimizers,
    )
