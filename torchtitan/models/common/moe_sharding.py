# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config-based sharding helpers for MoE submodules."""

import spmd_types as spmd
from spmd_types import SpmdType

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.base.sharding import ShardingConfig

from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
    residual_activation_placement,
    token_id_placement,
)


DP = MeshAxisName.DP
DP_REPLICATE = MeshAxisName.DP_REPLICATE
CP = MeshAxisName.CP
TP = MeshAxisName.TP
EP = MeshAxisName.EP
EDP_SHARD = MeshAxisName.EDP_SHARD


def replicated_param_placement_sparse() -> SpmdType:
    """Sparse-family placement for state shared across routed experts."""
    return SpmdType(
        {
            DP_REPLICATE: spmd.R,
            EDP_SHARD: spmd.R,
            EP: spmd.R,
        }
    )


def expert_param_placement_sparse() -> SpmdType:
    """Sparse-family placement for routed-expert weights (EP enabled).

    Insertion order matches canonical mesh order ``dp_replicate -> edp_shard ->
    ep`` so ``_needed_axes``'s first-insertion axis order resolves
    to the sparse_mesh.

    ``dp_replicate`` and ``edp_shard`` are FSDP storage axes: ``Replicate`` at
    ``distribute_tensor`` time, FSDP reshards ``edp_shard`` post-parallelize.
    EP always shards on dim 0 (the expert dim of ``(num_experts, *, *)``
    weights).
    """
    return SpmdType(
        {
            DP_REPLICATE: spmd.R,
            EDP_SHARD: spmd.R,
            EP: spmd.S(0),
        }
    )


def _tokens_per_expert_placement() -> SpmdType:
    """Placement for the ``tokens_per_expert_E`` buffer.

    Each DP/CP rank processes different data and accumulates partial token
    counts, so DP/CP axes are ``Partial``. MoE reuses the mesh axis named TP
    for sequence-token sharding under EP, so TP is also ``Partial``. When EP
    is disabled, model validation requires TP to have size 1 and this placement
    is filtered out at runtime.
    """
    return SpmdType(
        {
            DP: spmd.P,
            CP: spmd.P,
            TP: spmd.P,
        }
    )


def router_config() -> ShardingConfig:
    """Router input contracts and expert-count buffer placement.

    The padding mask follows ``x_TD`` at the MoE and Router boundaries. Under
    EP, the enclosing MoE sequence-shards both inputs before calling the Router.

    Under EP, input is Shard(0) on tokens, the gate computes on the local shard,
    and output remains Shard(0). Without EP, model validation requires TP to
    have size 1 and the TP placement is filtered out at runtime.
    """
    return ShardingConfig(
        state_shardings={
            "tokens_per_expert_E": _tokens_per_expert_placement(),
        },
        in_src_shardings={
            "x_TD": dense_sequence_parallel_placement(),
            "padding_mask_T": token_id_placement(enable_sp=True),
        },
    )


def router_gate_config() -> ShardingConfig:
    """Replicate the router gate parameters across TP."""
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.R),
            "bias": dense_param_placement(tp=spmd.R),
        },
    )


def shared_expert_rowwise_config(*, output_layout: SpmdType) -> ShardingConfig:
    """Shard shared-expert w2 and declare its explicit output layout."""
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.S(1)),
            "bias": dense_param_placement(tp=spmd.R),
        },
        in_src_shardings={
            "input": dense_activation_placement(tp=spmd.S(-1), cp=spmd.S(0))
        },
        out_src_shardings=output_layout,
    )


def shared_experts_output_placement(*, enable_sp: bool) -> SpmdType:
    """Output of the shared experts, combined with the routed output.

    Without SP the shared w2 output stays TP-Partial so the MoE reduces the
    routed and shared sum once.
    """
    if enable_sp:
        return dense_sequence_parallel_placement()
    return dense_activation_placement(tp=spmd.P, cp=spmd.S(0))


def shared_experts_config(*, enable_sp: bool) -> ShardingConfig:
    """Shared FeedForward boundary inside an MoE."""
    return ShardingConfig(
        in_src_shardings={"x": residual_activation_placement(enable_sp=enable_sp)},
        out_src_shardings=shared_experts_output_placement(enable_sp=enable_sp),
    )


def routed_experts_config() -> ShardingConfig:
    """Local-SPMD region of the routed experts.

    EP requires dense TP to shard routed tokens. Without EP, validation also
    requires TP to have size 1, so this TP placement is a runtime no-op.
    """
    experts_input_layout = dense_sequence_parallel_placement()
    return ShardingConfig(
        in_src_shardings={
            "x_TD": experts_input_layout,
            "topk_scores_TK": experts_input_layout,
            "topk_expert_ids_TK": experts_input_layout,
            "num_local_tokens_per_expert_E": _tokens_per_expert_placement(),
        },
        out_src_shardings=experts_input_layout,
        local_spmd=True,
    )


def routed_expert_weight_placement(*, enable_ep: bool) -> SpmdType:
    """Placement of routed-expert weights.

    Sparse ``{EP}`` placement when EP is enabled. Otherwise the weights are
    replicated on the dense mesh so FSDP can still shard them.
    """
    if enable_ep:
        return expert_param_placement_sparse()
    return dense_param_placement(tp=spmd.R)


def routed_expert_linear_config(*, enable_ep: bool) -> ShardingConfig:
    """Routed-expert ``GroupedLinear`` (``w13``/``w2``) weight sharding."""
    return ShardingConfig(
        state_shardings={"weight": routed_expert_weight_placement(enable_ep=enable_ep)}
    )


def moe_config(*, enable_sp: bool) -> ShardingConfig:
    """Input/output contracts for the MoE TP boundary.

    Rules are populated regardless of whether TP is enabled; ``resolve_mesh``
    filters out disabled axes at runtime.
    """
    layout = residual_activation_placement(enable_sp=enable_sp)
    return ShardingConfig(
        state_shardings={
            "expert_bias_E": dense_param_placement(tp=spmd.R),
        },
        in_src_shardings={
            "x_TD": layout,
            "padding_mask_T": token_id_placement(),
        },
        out_src_shardings=layout,
    )
