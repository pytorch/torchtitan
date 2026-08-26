# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Expert-parallel sharding for Kimi K3 routed experts."""

from torchtitan.models.common.moe_sharding import expert_param_placement_sparse
from torchtitan.protocols.sharding import ShardingConfig


_GROUPED_EXPERT_PARAM_NAMES = ("w1_EFD", "w2_EDF", "w3_EFD")


def set_kimi_k3_ep_sharding_config(config) -> None:
    """Shard routed-expert weights on the sparse EP/eFSDP mesh.

    Kimi K3 currently supports EP without TP or CP, so the surrounding decoder
    continues to operate on rank-local tensors. ``RoutedExperts.parallelize``
    wires the EP mesh into the dispatcher, while only the grouped-expert
    parameters are converted to sparse-mesh DTensors.
    """
    for layer_cfg in config.layers:
        if layer_cfg.moe is None:
            continue
        inner_experts = layer_cfg.moe.routed_experts.inner_experts
        inner_experts.sharding_config = ShardingConfig(
            state_shardings={
                name: expert_param_placement_sparse()
                for name in _GROUPED_EXPERT_PARAM_NAMES
            }
        )
