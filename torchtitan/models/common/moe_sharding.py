# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config-based sharding helpers for MoE submodules."""

import spmd_types as spmd

from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
)
from torchtitan.protocols.sharding import LocalMapConfig, NamedPlacement, ShardingConfig
from torchtitan.protocols.types import MeshAxisName


DP_REPLICATE = MeshAxisName.DP_REPLICATE
DP = MeshAxisName.DP
CP = MeshAxisName.CP
TP = MeshAxisName.TP
EP = MeshAxisName.EP
EFSDP = MeshAxisName.EFSDP


def expert_param_placement_sparse() -> NamedPlacement:
    """Sparse-family placement for routed-expert weights (EP enabled).

    Insertion order matches canonical mesh order ``DP_REPLICATE -> EFSDP ->
    EP`` so ``_needed_axes``'s first-insertion axis order resolves
    to the sparse_mesh.

    DP_REPLICATE / EFSDP are FSDP storage axes: ``R`` at
    ``distribute_tensor`` time, FSDP reshards ``EFSDP`` post-parallelize.
    EP always shards on dim 0 (the expert dim of ``(num_experts, *, *)``
    weights).
    """
    return NamedPlacement(
        {
            DP_REPLICATE: spmd.R,
            EFSDP: spmd.R,
            EP: spmd.S(0),
        }
    )


def expert_param_placement_dense(
    *, tp_placement: spmd.PerMeshAxisSpmdType
) -> NamedPlacement:
    """Dense-family placement for routed-expert weights (EP disabled, TP > 1).

    Used when expert parallelism is disabled but tensor parallelism shards
    the experts on the dense TP axis.

    Insertion order matches canonical mesh order. ``tp_placement`` is the
    placement on the TP axis: ``S(1)`` for colwise, ``S(2)`` for
    rowwise, ``R`` for replicated bias.
    """
    return NamedPlacement(
        {
            DP: spmd.R,
            CP: spmd.R,
            TP: tp_placement,
        }
    )


def _shared_expert_colwise_config(enable_ep: bool, enable_sp: bool) -> ShardingConfig:
    """Colwise shared-expert FFN (w1/w3).

    Mirrors ``ColwiseParallel(input_layouts=...)``: input is all-gathered
    to ``R`` for the column-sharded matmul; output is ``S(2)``
    (feature dim for 3-D activations from MoE).
    """
    sp_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I)
    )
    input_layout = dense_activation_placement(tp=spmd.R) if not enable_ep else sp_layout

    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.S(0)),
            "bias": dense_param_placement(tp=spmd.S(0)),
        },
        in_src_shardings={"input": input_layout},
        in_dst_shardings={"input": dense_activation_placement(tp=spmd.R)},
        out_src_shardings=dense_activation_placement(tp=spmd.S(2)),
        out_dst_shardings=dense_activation_placement(tp=spmd.S(2)),
    )


def _shared_expert_rowwise_config() -> ShardingConfig:
    """Rowwise shared-expert FFN (w2), output stays ``P``.

    Mirrors rowwise parallel with local output disabled:
    input ``S(2)`` (feature dim of 3-D activation) from upstream colwise;
    rowwise matmul produces ``P``; output stays ``P`` — reduction
    happens at the MoE boundary.
    """
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.S(1)),
            # Rowwise bias is R — addmm implicitly converts to P to match
            # the rowwise matmul output placement.
            "bias": dense_param_placement(tp=spmd.R),
        },
        in_src_shardings={"input": dense_activation_placement(tp=spmd.S(2))},
        out_src_shardings=dense_activation_placement(tp=spmd.P),
    )


def _router_gate_config(*, enable_ep: bool, enable_sp: bool) -> ShardingConfig:
    """Router gate: ``R`` weights, output stays typed.

    EP off: input ``R``, gate computes on all tokens, output ``R``.
    EP on:  input ``S(1)`` (slen dim of 3-D activation), gate computes on
            local shard, output ``S(1)``.
    """
    state = {
        "weight": dense_param_placement(tp=spmd.R),
        "bias": dense_param_placement(tp=spmd.R),
    }

    if not enable_ep:
        return ShardingConfig(state_shardings=state)

    input_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I)
    )
    routed_layout = dense_sequence_parallel_placement()
    return ShardingConfig(
        state_shardings=state,
        in_src_shardings={"input": input_layout},
        in_dst_shardings={"input": routed_layout},
        out_src_shardings=routed_layout,
        out_dst_shardings=routed_layout,
    )


def _tokens_per_expert_placement(*, enable_ep: bool) -> NamedPlacement:
    """Placement for the ``tokens_per_expert_E`` buffer.

    Each DP/CP rank processes different data and accumulates partial token
    counts, so DP/CP axes are ``P``. TP is ``P`` when EP is
    enabled (TP axis doubles as SP, each rank sees different tokens) or
    ``R`` when EP is disabled (all TP ranks see the same tokens).
    """
    tp_placement = spmd.P if enable_ep else spmd.R
    return NamedPlacement(
        {
            DP: spmd.P,
            CP: spmd.P,
            TP: tp_placement,
        }
    )


def _moe_sharding_config(*, enable_ep: bool, enable_sp: bool) -> ShardingConfig:
    """``ShardingConfig`` at the MoE boundary.

    Input arrives at sp_layout, redistributed to desired_input_layouts;
    output is ``P``, redistributed to sp_layout. MoE.forward()
    operates on typed tensors — the typed→local conversion happens at
    the GroupedExperts boundary.
    """
    sp_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I)
    )
    moe_desired_input_layouts = (
        dense_activation_placement(tp=spmd.R)
        if enable_sp and not enable_ep
        else sp_layout
    )
    return ShardingConfig(
        state_shardings={
            "expert_bias_E": dense_param_placement(tp=spmd.R),
            "tokens_per_expert_E": _tokens_per_expert_placement(enable_ep=enable_ep),
        },
        in_src_shardings={"x_BLD": sp_layout},
        in_dst_shardings={"x_BLD": moe_desired_input_layouts},
        out_src_shardings=dense_activation_placement(tp=spmd.P),
        out_dst_shardings=sp_layout,
    )


def set_moe_sharding_config(
    moe_cfg,
    *,
    enable_ep: bool,
    enable_sp: bool,
    expert_param_layout: dict[str, spmd.PerMeshAxisSpmdType],
) -> None:
    """Populate ``sharding_config`` on every MoE submodule.

    Branches dense vs sparse family per Module:

    - ``moe`` (wrapper): input/output redistribution on ``{TP}``.
      Always set when ``tp_enabled``.
    - ``moe.router.gate``: ``R`` weights, output stays typed.
    - ``moe.shared_experts.{w1,w2,w3}``: dense-family TP plan with
      ``P``-flow grad annotations (when ``moe_cfg.shared_experts is not
      None``).
    - ``moe.experts`` (``GroupedExperts``): sparse-family ``{EP}``
      placements when EP is enabled; dense-family ``{TP}`` placements
      when EP is disabled and TP is enabled; no sharding when both are
      disabled.

    ``expert_param_layout`` maps each routed-expert parameter name to its
    dense in/out-dim placement (used on the EP-disabled + TP-enabled path):
    ``S(1)`` for colwise, ``S(2)`` for rowwise, ``R`` for
    replicated bias. The shared ``GroupedExperts`` (qwen3, llama4,
    deepseek_v3) passes ``{"w1_EFD": S(1), "w2_EDF": S(2), "w3_EFD": S(1)}``;
    ``GptOssGroupedExperts`` passes its mlp1/mlp2 layout.

    Args:
        moe_cfg: The ``MoE.Config`` instance to populate.
        enable_ep: Whether expert parallelism is enabled.
        enable_sp: Whether sequence parallelism is enabled (affects the
            wrapper's enter/exit TP layout).
        expert_param_layout: ``{param_name: tp_placement}`` for the
            routed experts' weight params (used on the EP-disabled +
            TP-enabled path).
    """
    # Always set sharding configs regardless of whether TP is enabled.
    # ``resolve_mesh`` filters out disabled axes at runtime.
    moe_cfg.sharding_config = _moe_sharding_config(
        enable_ep=enable_ep, enable_sp=enable_sp
    )

    moe_cfg.router.gate.sharding_config = _router_gate_config(
        enable_ep=enable_ep,
        enable_sp=enable_sp,
    )

    # Shared experts: optional. Use P-flow variants so the
    # P->sp_layout reduce only happens once at the MoE boundary.
    if getattr(moe_cfg, "shared_experts", None) is not None:
        moe_cfg.shared_experts.w1.sharding_config = _shared_expert_colwise_config(
            enable_ep=enable_ep, enable_sp=enable_sp
        )
        moe_cfg.shared_experts.w2.sharding_config = _shared_expert_rowwise_config()
        moe_cfg.shared_experts.w3.sharding_config = _shared_expert_colwise_config(
            enable_ep=enable_ep, enable_sp=enable_sp
        )

    # Routed experts: local_map converts typed inputs to local for
    # dispatch/compute/combine, then wraps local output as P.
    # Routed experts: the three things that differ between EP and TP-only
    # are state_shardings, input layout, and input grad layout.
    experts_out_layout = dense_activation_placement(tp=spmd.P)
    if enable_ep:
        state_shardings: dict[str, NamedPlacement] = {
            name: expert_param_placement_sparse() for name in expert_param_layout
        }
        experts_in_layout = dense_sequence_parallel_placement()
        experts_in_grad_layout = experts_in_layout
        experts_x_src_layout = (
            experts_in_layout
            if enable_sp
            else dense_activation_placement(tp=spmd.I)
        )
    else:
        state_shardings = {
            name: expert_param_placement_dense(tp_placement=placement)
            for name, placement in expert_param_layout.items()
        }
        experts_in_layout = dense_activation_placement(tp=spmd.R)
        experts_in_grad_layout = dense_activation_placement(tp=spmd.P)
        experts_x_src_layout = dense_activation_placement(
            tp=spmd.R if enable_sp else spmd.I
        )

    moe_cfg.experts.sharding_config = ShardingConfig(
        state_shardings=state_shardings,
        in_src_shardings={
            "x_BLD": experts_x_src_layout,
            "topk_scores_BLK": experts_in_layout,
            "topk_expert_ids_BLK": experts_in_layout,
        },
        in_dst_shardings={
            "x_BLD": experts_in_layout,
            "topk_scores_BLK": experts_in_layout,
            "topk_expert_ids_BLK": experts_in_layout,
        },
        out_src_shardings=experts_out_layout,
        out_dst_shardings=experts_out_layout,
        local_map=LocalMapConfig(
            in_grad_placements=(
                experts_in_grad_layout,
                experts_in_grad_layout,
                experts_in_grad_layout,
            ),
        ),
    )
