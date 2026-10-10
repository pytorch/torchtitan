# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config-based sharding helpers for MoE submodules."""

import spmd_types as spmd
from spmd_types import SpmdType

from torchtitan.distributed.parallelism_context import MeshAxisName

from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
    stacked_colwise_config,
    token_id_placement,
)
from torchtitan.protocols.sharding import ShardingConfig


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


def _router_sharding_config() -> ShardingConfig:
    """Router input contracts and expert-count buffer placement.

    The padding mask follows ``x_TD`` at the MoE and Router boundaries. Under
    EP, the enclosing MoE sequence-shards both inputs before calling the Router.

    Under EP, input is Shard(0) on tokens, the gate computes on the local shard,
    and output remains Shard(0). Without EP, model validation requires TP to
    have size 1 and the TP placement is filtered out at runtime.
    """
    input_layout = dense_sequence_parallel_placement()
    padding_mask_layout = token_id_placement(enable_sp=True)
    return ShardingConfig(
        state_shardings={
            "tokens_per_expert_E": _tokens_per_expert_placement(),
            "routing_mismatch_counts": _tokens_per_expert_placement(),
        },
        in_src_shardings={
            "x_TD": input_layout,
            "padding_mask_T": padding_mask_layout,
            "routed_expert_ids_TK": token_id_placement(enable_sp=True, ndim=2),
        },
    )


def _router_gate_sharding_config() -> ShardingConfig:
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


def _shared_experts_sharding_configs(
    *, enable_sp: bool
) -> tuple[ShardingConfig, ShardingConfig, ShardingConfig]:
    """Configs for shared FeedForward parent and w13/w2 linears."""
    input_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
    )
    output_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.P, cp=spmd.S(0))
    )
    return (
        ShardingConfig(
            in_src_shardings={"x": input_layout},
            out_src_shardings=output_layout,
        ),
        stacked_colwise_config(input_layout=input_layout),
        shared_expert_rowwise_config(output_layout=output_layout),
    )


def _routed_experts_sharding_configs(
    *,
    enable_ep: bool,
) -> tuple[ShardingConfig, ShardingConfig, ShardingConfig]:
    """Configs for the routed local-SPMD region and grouped linears."""
    if enable_ep:
        w13_config = ShardingConfig(
            state_shardings={"weight": expert_param_placement_sparse()}
        )
        w2_config = ShardingConfig(
            state_shardings={"weight": expert_param_placement_sparse()}
        )
    else:
        # Expert weights still need SPMD annotations so FSDP can shard them on
        # the dense mesh.
        w13_config = ShardingConfig(
            state_shardings={"weight": dense_param_placement(tp=spmd.R)}
        )
        w2_config = ShardingConfig(
            state_shardings={"weight": dense_param_placement(tp=spmd.R)}
        )

    # EP requires dense TP to shard routed tokens. Without EP, validation also
    # requires TP to have size 1, so this TP placement is a runtime no-op.
    experts_input_layout = dense_sequence_parallel_placement()
    tokens_per_expert_layout = _tokens_per_expert_placement()
    return (
        ShardingConfig(
            in_src_shardings={
                "x_TD": experts_input_layout,
                "topk_scores_TK": experts_input_layout,
                "topk_expert_ids_TK": experts_input_layout,
                "num_local_tokens_per_expert_E": tokens_per_expert_layout,
            },
            out_src_shardings=experts_input_layout,
            local_spmd=True,
        ),
        w13_config,
        w2_config,
    )


def _moe_sharding_config(
    *,
    enable_sp: bool,
) -> ShardingConfig:
    """Input/output contracts for the MoE TP boundary."""
    sp_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
    )
    return ShardingConfig(
        state_shardings={
            "expert_bias_E": dense_param_placement(tp=spmd.R),
        },
        in_src_shardings={
            "x_TD": sp_layout,
            "padding_mask_T": token_id_placement(),
            "routed_expert_ids_TK": token_id_placement(ndim=2),
        },
        out_src_shardings=sp_layout,
    )


def set_routed_moe_sharding_config(
    moe_cfg,
    *,
    enable_ep: bool,
    enable_sp: bool,
) -> None:
    """Configure an MoE wrapper, router, and routed experts.

    Configures sparse expert parallelism when EP is enabled and leaves routed
    experts unsharded otherwise:

    - ``moe`` (wrapper): external input/output contracts on ``{TP}``.
    - ``moe.router``: input contracts plus the expert-count buffer placement.
    - ``moe.router.gate``: Replicate weights and output.
    - ``moe.routed_experts.{w13,w2}``: expert weights use sparse ``{EP}``
      placements when EP is enabled and dense replicated placements otherwise,
      so FSDP can shard them on the dense mesh. The parent owns the local-SPMD
      boundary.

    Args:
        moe_cfg: The ``MoE.Config`` instance to populate.
        enable_ep: Whether expert parallelism is enabled.
        enable_sp: Whether sequence parallelism is enabled (affects the
            wrapper's enter/exit TP layout).
    """
    # Always set sharding configs regardless of whether TP is enabled.
    # ``resolve_mesh`` filters out disabled axes at runtime.
    moe_cfg.sharding_config = _moe_sharding_config(enable_sp=enable_sp)
    moe_cfg.router.sharding_config = _router_sharding_config()

    moe_cfg.router.gate.sharding_config = _router_gate_sharding_config()

    # The RoutedExperts parent owns the local activation region. Its w13/w2
    # children own sparse expert state; other stateful children keep their own
    # sharding configs.
    routed_experts_config, w13_config, w2_config = _routed_experts_sharding_configs(
        enable_ep=enable_ep,
    )
    moe_cfg.routed_experts.sharding_config = routed_experts_config
    moe_cfg.routed_experts.w13.sharding_config = w13_config
    moe_cfg.routed_experts.w2.sharding_config = w2_config


def set_shared_moe_sharding_config(
    shared_experts_cfg,
    *,
    enable_sp: bool,
) -> None:
    """Configure shared experts with an explicit row-parallel output."""
    shared_config, w13_config, w2_config = _shared_experts_sharding_configs(
        enable_sp=enable_sp
    )
    shared_experts_cfg.sharding_config = shared_config
    shared_experts_cfg.w13.sharding_config = w13_config
    shared_experts_cfg.w2.sharding_config = w2_config


def set_moe_sharding_config(
    moe_cfg,
    *,
    enable_ep: bool,
    enable_sp: bool,
) -> None:
    """Configure a standard MoE, including its shared FeedForward."""
    set_routed_moe_sharding_config(
        moe_cfg,
        enable_ep=enable_ep,
        enable_sp=enable_sp,
    )

    shared = moe_cfg.shared_experts
    if shared is not None:
        set_shared_moe_sharding_config(shared, enable_sp=enable_sp)
