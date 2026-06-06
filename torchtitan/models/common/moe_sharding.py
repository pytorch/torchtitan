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
)
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig, SpmdLayout
from torchtitan.protocols.types import MeshAxisName


DP = MeshAxisName.DP
DP_REPLICATE = MeshAxisName.DP_REPLICATE
CP = MeshAxisName.CP
TP = MeshAxisName.TP
EP = MeshAxisName.EP
EFSDP = MeshAxisName.EFSDP


def expert_param_placement_sparse() -> SpmdLayout:
    """Sparse-family placement for routed-expert weights (EP enabled).

    Insertion order matches canonical mesh order ``DP_REPLICATE -> EFSDP ->
    EP`` so ``_needed_axes``'s first-insertion axis order resolves
    to the sparse_mesh.

    DP_REPLICATE / EFSDP are FSDP storage axes: ``Replicate`` at
    ``distribute_tensor`` time, FSDP reshards ``EFSDP`` post-parallelize.
    EP always shards on dim 0 (the expert dim of ``(num_experts, *, *)``
    weights).
    """
    return SpmdLayout(
        {
            DP_REPLICATE: spmd.R,
            EFSDP: spmd.R,
            EP: spmd.S(0),
        }
    )


def expert_param_placement_dense(
    *, tp_placement: spmd.PerMeshAxisSpmdType
) -> SpmdLayout:
    """Dense-family placement for routed-expert weights (EP disabled, TP > 1).

    Used when expert parallelism is disabled but tensor parallelism shards
    the experts on the dense TP axis.

    Insertion order matches canonical mesh order. ``tp_placement`` is the
    placement on the TP axis: ``Shard(1)`` for colwise, ``Shard(2)`` for
    rowwise, ``Replicate()`` for replicated bias.
    """
    return SpmdLayout(
        {
            DP: spmd.R,
            CP: spmd.R,
            TP: tp_placement,
        }
    )


def _shared_expert_colwise_config(enable_ep: bool, enable_sp: bool) -> ShardingConfig:
    """Colwise shared-expert FFN (w1/w3).

    Mirrors ``ColwiseParallel(input_layouts=...)``: input is all-gathered
    to Replicate for the column-sharded matmul; output is Shard(2)
    (feature dim for 3-D activations from MoE).
    """
    sp_layout = spmd.S(1) if enable_sp else spmd.R
    input_layout = spmd.R if not enable_ep else sp_layout
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.S(0)),
            "bias": dense_param_placement(tp=spmd.S(0)),
        },
        in_src_shardings={"input": dense_activation_placement(tp=input_layout)},
        in_dst_shardings={"input": dense_activation_placement(tp=spmd.R)},
        out_dst_shardings=dense_activation_placement(tp=spmd.S(2)),
    )


def _shared_expert_rowwise_config() -> ShardingConfig:
    """Rowwise shared-expert FFN (w2), output stays DTensor(Partial).

    Mirrors ``RowwiseParallel(output_layouts=Partial(), use_local_output=False)``:
    input Shard(2) (feature dim of 3-D activation) from upstream colwise;
    rowwise matmul produces Partial; output stays DTensor(Partial) — reduction
    happens at the MoE boundary.
    """
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.S(1)),
            # Rowwise bias is Replicate — addmm implicitly converts to
            # Partial to match the rowwise matmul output placement.
            "bias": dense_param_placement(tp=spmd.R),
        },
        in_src_shardings={"input": dense_activation_placement(tp=spmd.S(2))},
        out_dst_shardings=dense_activation_placement(tp=spmd.P),
    )


def _router_gate_config(*, enable_ep: bool) -> ShardingConfig:
    """Router gate: Replicate weights, output stays DTensor.

    EP off: input Replicate, gate computes on all tokens, output DTensor(Replicate).
    EP on:  input Shard(1) (slen dim of 3-D activation), gate computes on
            local shard, output DTensor(Shard(1)).
    """
    state = {
        "weight": dense_param_placement(tp=spmd.R),
        "bias": dense_param_placement(tp=spmd.R),
    }
    if enable_ep:
        return ShardingConfig(
            state_shardings=state,
            in_dst_shardings={"input": dense_activation_placement(tp=spmd.S(1))},
            out_dst_shardings=dense_activation_placement(tp=spmd.S(1)),
        )
    else:
        return ShardingConfig(
            state_shardings=state,
            in_dst_shardings={"input": dense_activation_placement(tp=spmd.R)},
            out_dst_shardings=dense_activation_placement(tp=spmd.R),
        )


def _tokens_per_expert_placement(*, enable_ep: bool) -> SpmdLayout:
    """Placement for the ``tokens_per_expert_E`` buffer.

    Each DP/CP rank processes different data and accumulates partial token
    counts, so DP/CP axes are ``Partial``. TP is ``Partial`` when EP is
    enabled (TP axis doubles as SP, each rank sees different tokens) or
    ``Replicate`` when EP is disabled (all TP ranks see the same tokens).
    """
    return SpmdLayout(
        {
            DP: spmd.P,
            CP: spmd.P,
            TP: spmd.P if enable_ep else spmd.R,
        }
    )


def _moe_sharding_config(*, enable_ep: bool, enable_sp: bool) -> ShardingConfig:
    """``ShardingConfig`` at the MoE boundary.

    Input arrives at sp_layout, redistributed to desired_input_layouts;
    output is Partial, redistributed to sp_layout. MoE.forward()
    operates on DTensors — the DTensor→local conversion happens at
    the GroupedExperts boundary.
    """
    sp_layout = spmd.S(1) if enable_sp else spmd.R
    moe_desired_input_layouts = spmd.R if not enable_ep else sp_layout

    return ShardingConfig(
        state_shardings={
            "expert_bias_E": dense_param_placement(tp=spmd.R),
            "tokens_per_expert_E": _tokens_per_expert_placement(enable_ep=enable_ep),
        },
        in_src_shardings={"x_BLD": dense_activation_placement(tp=sp_layout)},
        in_dst_shardings={
            "x_BLD": dense_activation_placement(tp=moe_desired_input_layouts)
        },
        out_src_shardings=dense_activation_placement(tp=spmd.P),
        out_dst_shardings=dense_activation_placement(tp=sp_layout),
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
    - ``moe.router.gate``: Replicate weights, output stays DTensor.
    - ``moe.shared_experts.{w1,w2,w3}``: dense-family TP plan with
      Partial-flow grad annotations (when ``moe_cfg.shared_experts is not
      None``).
    - ``moe.experts`` (``GroupedExperts``): sparse-family ``{EP}``
      placements when EP is enabled; dense-family ``{TP}`` placements
      when EP is disabled and TP is enabled; no sharding when both are
      disabled.

    ``expert_param_layout`` maps each routed-expert parameter name to its
    dense in/out-dim placement (used on the EP-disabled + TP-enabled path):
    ``Shard(1)`` for colwise, ``Shard(2)`` for rowwise, ``Replicate()`` for
    replicated bias. The shared ``GroupedExperts`` (qwen3, llama4,
    deepseek_v3) passes ``{"w1_EFD": Shard(1), "w2_EDF": Shard(2), "w3_EFD": Shard(1)}``;
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

    # Router gate: dense-family TP plan with Partial output grad.
    moe_cfg.router.gate.sharding_config = _router_gate_config(enable_ep=enable_ep)

    # Shared experts: optional. Use Partial-flow variants so the
    # Partial->sp_layout reduce only happens once at the MoE boundary.
    if getattr(moe_cfg, "shared_experts", None) is not None:
        moe_cfg.shared_experts.w1.sharding_config = _shared_expert_colwise_config(
            enable_ep=enable_ep, enable_sp=enable_sp
        )
        moe_cfg.shared_experts.w2.sharding_config = _shared_expert_rowwise_config()
        moe_cfg.shared_experts.w3.sharding_config = _shared_expert_colwise_config(
            enable_ep=enable_ep, enable_sp=enable_sp
        )

    # Routed experts: local_map converts DTensor inputs to local for
    # dispatch/compute/combine, then wraps local output as DTensor(Partial).
    # Routed experts: the three things that differ between EP and TP-only
    # are state_shardings, input layout, and input grad layout.
    experts_out_layout = dense_activation_placement(tp=spmd.P)
    if enable_ep:
        state_shardings: dict[str, SpmdLayout] = {
            name: expert_param_placement_sparse() for name in expert_param_layout
        }
        experts_in_layout = dense_activation_placement(tp=spmd.S(1))
        experts_in_grad_layout = dense_activation_placement(tp=spmd.S(1))
    else:
        state_shardings = {
            name: expert_param_placement_dense(tp_placement=placement)
            for name, placement in expert_param_layout.items()
        }
        experts_in_layout = dense_activation_placement(tp=spmd.R)
        experts_in_grad_layout = dense_activation_placement(tp=spmd.P)

    moe_cfg.experts.sharding_config = ShardingConfig(
        state_shardings=state_shardings,
        in_dst_shardings={
            "x_BLD": experts_in_layout,
            "topk_scores_BLK": experts_in_layout,
            "topk_expert_ids_BLK": experts_in_layout,
            "num_local_tokens_per_expert_E": _tokens_per_expert_placement(
                enable_ep=enable_ep
            ),
        },
        out_src_shardings=experts_out_layout,
        out_dst_shardings=experts_out_layout,
        local_map=LocalMapConfig(
            in_grad_placements=(
                experts_in_grad_layout,
                experts_in_grad_layout,
                experts_in_grad_layout,
                # num_local_tokens_per_expert_E is routing metadata, but it is
                # still a DTensor input to local_map and must have placements.
                _tokens_per_expert_placement(enable_ep=enable_ep),
            ),
        ),
    )
