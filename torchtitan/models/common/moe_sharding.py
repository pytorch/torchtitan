# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config-based sharding helpers for MoE submodules."""

import spmd_types as spmd
from spmd_types import SpmdType

from torchtitan.distributed.parallel_dims import MeshAxisName

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
EFSDP = MeshAxisName.EFSDP

_GROUPED_EXPERT_PARAM_NAMES = ("w1_EFD", "w2_EDF", "w3_EFD")


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


def _tokens_per_expert_placement(*, enable_ep: bool) -> SpmdType:
    """Placement for the ``tokens_per_expert_E`` buffer.

    Each DP/CP rank processes different data and accumulates partial token
    counts, so DP/CP axes are ``Partial``. TP is ``Partial`` when EP is
    enabled (MoE reuses the mesh axis named TP for sequence-token sharding, so
    each rank sees different tokens).
    """
    return SpmdType(
        {
            DP: spmd.P,
            CP: spmd.P,
            TP: spmd.P if enable_ep else spmd.R,
        }
    )


def _router_sharding_config(*, enable_ep: bool, enable_sp: bool) -> ShardingConfig:
    """Router input redistribution and expert-count buffer placement.

    The padding mask follows ``x_TD`` at the MoE and Router boundaries. Under
    EP, the Router then sequence-shards both inputs before routing.

    EP off: input Replicate, gate computes on all tokens, output stays Replicate.
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
    else:
        input_layout = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
        desired_input_layout = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))

    padding_mask_layout = token_id_placement(enable_sp=enable_sp and enable_ep)
    desired_padding_mask_layout = token_id_placement(enable_sp=enable_ep)
    return ShardingConfig(
        state_shardings={
            "tokens_per_expert_E": _tokens_per_expert_placement(enable_ep=enable_ep),
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
    return stacked_colwise_config()


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
    # w2 reduces its Partial output to the final MoE boundary layout
    # used for the routed + shared add: sequence-sharded when SP is enabled and
    # Partial when SP is disabled.
    input_layout = (
        dense_sequence_parallel_placement()
        if enable_ep and enable_sp
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
) -> tuple[ShardingConfig, ShardingConfig | None]:
    """Configs for RoutedExperts local SPMD and inner expert weight state."""
    if enable_ep:
        pre_experts_input_layout = (
            dense_sequence_parallel_placement()
            if enable_sp
            else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
        )
        state_shardings: dict[str, SpmdType] = {
            name: expert_param_placement_sparse()
            for name in _GROUPED_EXPERT_PARAM_NAMES
        }
        experts_input_layout = dense_sequence_parallel_placement()
        inner_experts_sharding_config = ShardingConfig(state_shardings=state_shardings)
    else:
        pre_experts_input_layout = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
        experts_input_layout = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
        inner_experts_sharding_config = None

    tokens_per_expert_layout = _tokens_per_expert_placement(enable_ep=enable_ep)

    experts_output_layout = (
        dense_sequence_parallel_placement()
        if enable_ep
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
        inner_experts_sharding_config,
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
        sp_layout if enable_ep else dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
    )
    output_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(
            tp=spmd.P if enable_ep else spmd.R, cp=spmd.S(0)
        )
    )
    padding_mask_src_layout = token_id_placement(enable_sp=enable_sp)
    padding_mask_dst_layout = token_id_placement(enable_sp=enable_sp and enable_ep)
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
) -> None:
    """Populate ``sharding_config`` on every MoE submodule.

    Configures sparse expert parallelism when EP is enabled and leaves routed
    experts unsharded otherwise:

    - ``moe`` (wrapper): input/output redistribution on ``{TP}``.
    - ``moe.router``: input and padding-mask redistribution to the router's
      token layout, plus the expert-count buffer placement.
    - ``moe.router.gate``: Replicate weights and output.
    - ``moe.shared_experts.{w13,w2}``: dense-family TP plan (when
      ``moe_cfg.shared_experts is not None``).
    - ``moe.routed_experts.inner_experts`` (``GroupedExperts``): expert-weight
      ``state_shardings`` -- sparse ``{EP}`` when EP is enabled and unsharded
      otherwise. The parent ``routed_experts`` holds the activation shardings
      and local SPMD region.

    Args:
        moe_cfg: The ``MoE.Config`` instance to populate.
        enable_ep: Whether expert parallelism is enabled.
        enable_sp: Whether sequence parallelism is enabled (affects the
            wrapper's enter/exit TP layout).
    """
    # Always set sharding configs regardless of whether TP is enabled.
    # ``resolve_mesh`` filters out disabled axes at runtime.
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
    (
        routed_experts_sharding_config,
        inner_experts_sharding_config,
    ) = _routed_experts_sharding_configs(
        enable_ep=enable_ep,
        enable_sp=enable_sp,
    )
    moe_cfg.routed_experts.sharding_config = routed_experts_sharding_config
    moe_cfg.routed_experts.inner_experts.sharding_config = inner_experts_sharding_config


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
