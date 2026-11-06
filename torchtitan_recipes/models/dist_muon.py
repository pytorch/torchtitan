# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared DistMuon compute-layout policies for Kimi recipes."""

from dataclasses import replace
from typing import cast

import torch
from torch.distributed.tensor import Shard

from torchtitan.components.optim import DistMuon, OptimizersContainer
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.flex_shard import BucketConfig, ComputeLayout
from torchtitan.distributed.parallelism_context import MeshAxisName


def per_expert_compute_layout(parallelism: ParallelismConfig) -> ComputeLayout:
    """Build BF16 expert ownership from the recipe's dense or sparse mesh."""
    ep_size = parallelism.expert_parallel_degree
    if ep_size <= 0:
        raise ValueError("expert_parallel_degree must be positive")
    # Routed experts use TP only without EP. With EP enabled, their sparse
    # mesh contains edp_shard and ep instead of tp.
    if ep_size == 1:
        # FSDP flattens dp_shard and cp into one storage axis under CP.
        shardings = {
            MeshAxisName.DP_SHARD.value: Shard(0),
            f"{MeshAxisName.DP_SHARD.value}_{MeshAxisName.CP.value}": Shard(0),
        }
        if parallelism.tensor_parallel_degree > 1:
            shardings[MeshAxisName.TP.value] = Shard(0)
        return ComputeLayout(
            shardings_by_mesh_axis=shardings,
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


def align_dist_muon_compute_layouts(
    optimizer_config: OptimizersContainer.Config,
    *,
    compute_config: tuple[dict[str, ComputeLayout], tuple[BucketConfig, ...], str],
) -> OptimizersContainer.Config:
    """Align Kimi Muon layouts and buckets without mutating optimizer settings.

    Config variants replace model/parallelism through ``dataclasses.replace``.
    Their final compute config controls TP head ownership, transport buckets,
    and the routed-expert mesh; preserve all optimizer hyperparameters.
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
    compute_sharding_by_fqn, bucket_configs, pattern = compute_config
    if (
        dist_muon_config.compute_sharding_by_fqn == compute_sharding_by_fqn
        and dist_muon_config.bucket_configs == bucket_configs
        and dist_muon_config.pattern == pattern
    ):
        return optimizer_config

    optimizers = list(optimizer_config.optimizers)
    optimizers[dist_muon_index] = replace(
        dist_muon_config,
        compute_sharding_by_fqn=compute_sharding_by_fqn,
        bucket_configs=bucket_configs,
        pattern=pattern,
    )
    return replace(
        optimizer_config,
        optimizers=optimizers,
    )
