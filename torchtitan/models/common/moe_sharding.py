# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config-based sharding helpers for MoE submodules."""

from collections.abc import Collection

import spmd_types as spmd
from spmd_types import SpmdType

from torchtitan.distributed.parallel_dims import MeshAxisName

from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
    token_id_placement,
)
from torchtitan.models.common.moe import MicrobatchWiseLoadBalanceLoss
from torchtitan.protocols.sharding import ShardingConfig


DP = MeshAxisName.DP
DP_REPLICATE = MeshAxisName.DP_REPLICATE
CP = MeshAxisName.CP
TP = MeshAxisName.TP
EP = MeshAxisName.EP
EFSDP = MeshAxisName.EFSDP


def expert_param_placement_sparse() -> SpmdType:
    """Sparse-family placement for routed-expert weights (EP enabled).

    Insertion order matches canonical mesh order ``DP_REPLICATE -> EFSDP ->
    EP`` so ``_needed_axes``'s first-insertion axis order resolves
    to the sparse_mesh.

    DP_REPLICATE / EFSDP are FSDP storage axes: ``Replicate`` at
    ``distribute_tensor`` time, FSDP reshards ``EFSDP`` post-parallelize.
    EP always shards on dim 0 (the expert dim of ``(num_experts, *, *)``
    weights).
    """
    return SpmdType(
        {
            DP_REPLICATE: spmd.R,
            EFSDP: spmd.R,
            EP: spmd.S(0),
        }
    )


def _tokens_per_expert_placement(
    *, enable_ep: bool, enable_sp: bool = False
) -> SpmdType:
    """Placement for the ``tokens_per_expert_E`` buffer.

    Each DP/CP rank processes different data and accumulates partial token
    counts, so DP/CP axes are ``Partial``. TP is ``Partial`` whenever the
    router tokens are sharded across TP, through either EP or dense SP.
    """
    return SpmdType(
        {
            DP: spmd.P,
            CP: spmd.P,
            TP: spmd.P if enable_ep or enable_sp else spmd.R,
        }
    )


def _router_sharding_config(*, enable_ep: bool, enable_sp: bool) -> ShardingConfig:
    """Router input redistribution and expert-count buffer placement.

    The padding mask follows ``x_TD`` at the MoE and Router boundaries. Under
    EP, the Router sequence-shards both inputs before routing, even when dense
    SP is disabled.

    EP off, SP off: input Replicate, gate computes on all tokens, output stays
                    Replicate.
    EP off, SP on: input Shard(0) on tokens, gate computes on the local shard,
                   and the output remains Shard(0).
    EP on: input Shard(0) on tokens, gate computes on the local shard, and the
           output remains Shard(0).
    """
    if enable_ep:
        input_layout = (
            dense_sequence_parallel_placement()
            if enable_sp
            else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
        )
        desired_input_layout = dense_sequence_parallel_placement()
    elif enable_sp:
        input_layout = dense_sequence_parallel_placement()
        desired_input_layout = dense_sequence_parallel_placement()
    else:
        input_layout = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
        desired_input_layout = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))

    padding_mask_layout = token_id_placement(enable_sp=enable_sp)
    desired_padding_mask_layout = token_id_placement(enable_sp=enable_ep or enable_sp)
    return ShardingConfig(
        state_shardings={
            "tokens_per_expert_E": _tokens_per_expert_placement(
                enable_ep=enable_ep, enable_sp=enable_sp
            ),
        },
        in_src_shardings={
            "x_TD": input_layout,
            "padding_mask_T": padding_mask_layout,
        },
        in_dst_shardings={
            "x_TD": desired_input_layout,
            "padding_mask_T": desired_padding_mask_layout,
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


def _shared_expert_colwise_config() -> ShardingConfig:
    """Colwise shared-expert FFN (w13).

    Mirrors ``ColwiseParallel(input_layouts=...)``: input is all-gathered
    to Replicate for the column-sharded matmul; output is Shard(1) on features.
    """
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.S(0)),
            "bias": dense_param_placement(tp=spmd.S(0)),
        },
        in_src_shardings={"input": dense_activation_placement(tp=spmd.R, cp=spmd.S(0))},
        in_dst_shardings={"input": dense_activation_placement(tp=spmd.R, cp=spmd.S(0))},
        out_src_shardings=dense_activation_placement(tp=spmd.S(1), cp=spmd.S(0)),
        out_dst_shardings=dense_activation_placement(tp=spmd.S(1), cp=spmd.S(0)),
    )


def _shared_expert_rowwise_config(*, output_layout: SpmdType) -> ShardingConfig:
    """Rowwise shared-expert FFN (w2).

    Mirrors ``RowwiseParallel``: input is Shard(1) on the feature dim from
    upstream colwise; rowwise matmul produces Partial, then redistributes to
    ``output_layout``.
    """
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.S(1)),
            # Rowwise bias is Replicate; addmm implicitly converts to Partial
            # to match the rowwise matmul output placement.
            "bias": dense_param_placement(tp=spmd.R),
        },
        in_src_shardings={
            "input": dense_activation_placement(tp=spmd.S(1), cp=spmd.S(0))
        },
        out_src_shardings=dense_activation_placement(tp=spmd.P, cp=spmd.S(0)),
        out_dst_shardings=output_layout,
    )


def _shared_experts_sharding_configs(
    *,
    enable_ep: bool,
    enable_sp: bool,
) -> tuple[ShardingConfig, ShardingConfig, ShardingConfig]:
    """Configs for shared FeedForward parent and w13/w2 linears."""
    # The parent FeedForward converts its input to Replicate once before w13.
    # w2 reduces its Partial output to the final MoE boundary layout used for
    # the routed + shared add.
    input_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(
            tp=spmd.I if enable_ep else spmd.R, cp=spmd.S(0)
        )
    )
    desired_input_layout = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
    desired_output_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(
            tp=spmd.P if enable_ep else spmd.R, cp=spmd.S(0)
        )
    )
    return (
        ShardingConfig(
            in_src_shardings={"x": input_layout},
            in_dst_shardings={"x": desired_input_layout},
        ),
        _shared_expert_colwise_config(),
        _shared_expert_rowwise_config(output_layout=desired_output_layout),
    )


def _routed_experts_sharding_configs(
    *,
    enable_ep: bool,
    enable_sp: bool,
    expert_param_names: Collection[str],
) -> tuple[ShardingConfig, ShardingConfig]:
    """Configs for RoutedExperts local SPMD and inner expert weight state."""
    if enable_ep:
        pre_experts_input_layout = (
            dense_sequence_parallel_placement()
            if enable_sp
            else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
        )
        state_shardings: dict[str, SpmdType] = {
            name: expert_param_placement_sparse() for name in expert_param_names
        }
        experts_input_layout = dense_sequence_parallel_placement()
    else:
        pre_experts_input_layout = (
            dense_sequence_parallel_placement()
            if enable_sp
            else dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
        )
        state_shardings = {
            name: dense_param_placement(tp=spmd.R) for name in expert_param_names
        }
        experts_input_layout = pre_experts_input_layout

    tokens_per_expert_layout = _tokens_per_expert_placement(
        enable_ep=enable_ep, enable_sp=enable_sp
    )

    experts_output_layout = (
        dense_sequence_parallel_placement()
        if enable_ep or enable_sp
        else dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
    )
    desired_experts_output_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(
            tp=spmd.P if enable_ep else spmd.R, cp=spmd.S(0)
        )
    )

    return (
        ShardingConfig(
            in_src_shardings={
                "x_TD": pre_experts_input_layout,
                "topk_scores_TK": experts_input_layout,
                "topk_expert_ids_TK": experts_input_layout,
                "num_local_tokens_per_expert_E": tokens_per_expert_layout,
            },
            in_dst_shardings={
                "x_TD": experts_input_layout,
                "topk_scores_TK": experts_input_layout,
                "topk_expert_ids_TK": experts_input_layout,
                "num_local_tokens_per_expert_E": tokens_per_expert_layout,
            },
            out_src_shardings=experts_output_layout,
            out_dst_shardings=desired_experts_output_layout,
            local_spmd=True,
        ),
        ShardingConfig(state_shardings=state_shardings),
    )


def _moe_sharding_config(
    *,
    enable_ep: bool,
    enable_sp: bool,
) -> ShardingConfig:
    """``ShardingConfig`` at the MoE boundary.

    Input arrives at sp_layout and is redistributed to desired_input_layouts.
    Output is redistributed to sp_layout. RoutedExperts runs in a local SPMD
    region.
    """
    sp_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
    )
    desired_input_layout = (
        sp_layout
        if enable_ep or enable_sp
        else dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
    )
    output_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(
            tp=spmd.P if enable_ep else spmd.R, cp=spmd.S(0)
        )
    )
    padding_mask_src_layout = token_id_placement(enable_sp=enable_sp)
    padding_mask_dst_layout = token_id_placement(enable_sp=enable_sp)
    return ShardingConfig(
        state_shardings={
            "expert_bias_E": dense_param_placement(tp=spmd.R),
        },
        in_src_shardings={
            "x_TD": sp_layout,
            "padding_mask_T": padding_mask_src_layout,
        },
        in_dst_shardings={
            "x_TD": desired_input_layout,
            "padding_mask_T": padding_mask_dst_layout,
        },
        out_src_shardings=output_layout,
        out_dst_shardings=sp_layout,
    )


def set_moe_sharding_config(
    moe_cfg,
    *,
    enable_ep: bool,
    enable_sp: bool,
    expert_param_names: Collection[str],
) -> None:
    """Populate ``sharding_config`` on every MoE submodule.

    Configures sparse expert parallelism when EP is enabled and replicates the
    routed experts across the dense TP axis otherwise:

    - ``moe`` (wrapper): input/output redistribution on ``{TP}``.
      Always set when ``tp_enabled``.
    - ``moe.router``: input and padding-mask redistribution to the router's
      token layout, plus the expert-count buffer placement.
    - ``moe.router.gate``: Replicate weights and output.
    - ``moe.shared_experts.{w13,w2}``: dense-family TP plan (when
      ``moe_cfg.shared_experts is not None``).
    - ``moe.routed_experts.inner_experts`` (``GroupedExperts``): expert-weight
      ``state_shardings`` -- sparse ``{EP}`` when EP is enabled and replicated
      across ``{TP}`` otherwise. The parent ``routed_experts`` holds the
      activation shardings and local SPMD region.

    Args:
        moe_cfg: The ``MoE.Config`` instance to populate.
        enable_ep: Whether expert parallelism is enabled.
        enable_sp: Whether sequence parallelism is enabled (affects the
            wrapper's enter/exit TP layout).
        expert_param_names: Routed-expert parameter names.
    """
    # Always set sharding configs regardless of whether TP is enabled.
    # ``resolve_mesh`` filters out disabled axes at runtime.
    tp_shards_tokens = enable_ep or enable_sp
    moe_cfg.tp_shards_tokens = tp_shards_tokens
    aux_loss_cfg = moe_cfg.router.aux_loss
    if isinstance(aux_loss_cfg, MicrobatchWiseLoadBalanceLoss.Config):
        aux_loss_cfg.tp_shards_tokens = tp_shards_tokens

    moe_cfg.sharding_config = _moe_sharding_config(
        enable_ep=enable_ep,
        enable_sp=enable_sp,
    )
    moe_cfg.router.sharding_config = _router_sharding_config(
        enable_ep=enable_ep,
        enable_sp=enable_sp,
    )

    moe_cfg.router.gate.sharding_config = _router_gate_sharding_config()

    # Shared experts: SwiGLU FFN run in parallel with the routed experts.
    shared = moe_cfg.shared_experts
    if shared is not None:
        (shared_config, w13_config, w2_config,) = _shared_experts_sharding_configs(
            enable_ep=enable_ep,
            enable_sp=enable_sp,
        )
        shared.sharding_config = shared_config
        shared.w13.sharding_config = w13_config
        shared.w2.sharding_config = w2_config

    # RoutedExperts local SPMD region: activation in/out, no params.
    routed_experts_config, inner_experts_config = _routed_experts_sharding_configs(
        enable_ep=enable_ep,
        enable_sp=enable_sp,
        expert_param_names=expert_param_names,
    )
    moe_cfg.routed_experts.sharding_config = routed_experts_config
    moe_cfg.routed_experts.inner_experts.sharding_config = inner_experts_config


def set_moe_block_padding_mask_sharding(block_cfg, *, enable_sp: bool) -> None:
    """Configure a MoE block's padding-mask input sharding.

    The mask enters TP-replicated and follows the block activation's token
    layout when sequence parallelism is enabled.
    """
    sharding_config = block_cfg.sharding_config or ShardingConfig()
    sharding_config.in_src_shardings = {
        **(sharding_config.in_src_shardings or {}),
        "padding_mask": token_id_placement(),
    }
    sharding_config.in_dst_shardings = {
        **(sharding_config.in_dst_shardings or {}),
        "padding_mask": token_id_placement(enable_sp=enable_sp),
    }
    block_cfg.sharding_config = sharding_config
