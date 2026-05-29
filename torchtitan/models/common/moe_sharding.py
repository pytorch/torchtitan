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
    dense_sp_placement,
)
from torchtitan.protocols.sharding import NamedPlacement, ShardingConfig
from torchtitan.protocols.types import MeshAxisName


DP_REPLICATE = MeshAxisName.DP_REPLICATE
DP_SHARD = MeshAxisName.DP_SHARD
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

    DP_REPLICATE / EFSDP are FSDP storage axes: ``Replicate`` at
    ``distribute_tensor`` time, FSDP reshards ``EFSDP`` post-parallelize.
    EP always shards on dim 0 (the expert dim of ``(num_experts, *, *)``
    weights).
    """
    return {
        DP_REPLICATE: spmd.R,
        EFSDP: spmd.R,
        EP: spmd.S(0),
    }


def expert_param_placement_dense(
    *, tp_placement: spmd.PerMeshAxisSpmdType
) -> NamedPlacement:
    """Dense-family placement for routed-expert weights (EP disabled, TP > 1).

    Used when expert parallelism is disabled but tensor parallelism shards
    the experts on the dense TP axis.

    ``tp_placement`` is the placement on the TP axis: ``S(1)`` for colwise,
    ``S(2)`` for rowwise, ``I`` for replicated bias.
    """
    return {
        DP: spmd.R,
        CP: spmd.R,
        TP: tp_placement,
    }


def _shared_expert_colwise_config(enable_ep: bool, enable_sp: bool) -> ShardingConfig:
    """Colwise shared-expert FFN (w1/w3).

    Mirrors ``ColwiseParallel(input_layouts=...)``: input is all-gathered
    to Replicate for the column-sharded matmul; output is Shard(2)
    (feature dim for 3-D activations from MoE).
    """
    if enable_ep and enable_sp:
        input_layout = dense_sp_placement()
    elif enable_sp:
        input_layout = dense_activation_placement(tp=spmd.R)
    else:
        input_layout = dense_activation_placement(tp=spmd.I)

    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.S(0)),
            "bias": dense_param_placement(tp=spmd.S(0)),
        },
        in_src_shardings={"input": input_layout},
        in_dst_shardings={"input": dense_activation_placement(tp=spmd.R)},
        out_dst_shardings=dense_activation_placement(tp=spmd.S(2)),
    )


def _shared_expert_rowwise_config() -> ShardingConfig:
    """Rowwise shared-expert FFN (w2), output stays partial.

    Input is feature-sharded from upstream colwise; rowwise matmul produces
    partial output. Reduction happens at the MoE boundary.
    """
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.S(1)),
            "bias": dense_param_placement(tp=spmd.I),
        },
        in_src_shardings={"input": dense_activation_placement(tp=spmd.S(2))},
        out_src_shardings=dense_activation_placement(tp=spmd.P),
    )


def _router_gate_config(
    *, enable_ep: bool, enable_sp: bool, has_bias: bool = False
) -> ShardingConfig:
    """Router gate state.

    Parameters are I->R@TP for SP, I@TP for no SP. No redistribution.
    """
    state = {
        "weight": dense_param_placement(tp=spmd.I),
    }
    compute_state = {}
    if enable_ep or enable_sp:
        compute_state = {
            "weight": dense_param_placement(tp=spmd.R),
        }
    if has_bias:
        state["bias"] = dense_param_placement(tp=spmd.I)
        if enable_ep or enable_sp:
            compute_state["bias"] = dense_param_placement(tp=spmd.R)

    return ShardingConfig(
        state_shardings=state,
        state_shardings_compute=compute_state,
    )


def _tokens_per_expert_placement(*, enable_ep: bool, enable_sp: bool) -> NamedPlacement:
    """Placement for the ``tokens_per_expert_E`` buffer.

    if EP on -> P@TP, as activations are sequence-split
    if EP off -> R@TP if SP on else I@TP (MoE boundary does a sequence-dim allgather; no split)
    """
    if enable_ep:
        tp_placement = spmd.P
    elif enable_sp:
        tp_placement = spmd.R
    else:
        tp_placement = spmd.I
    return {
        DP: spmd.P,
        CP: spmd.P,
        TP: tp_placement,
    }


def _moe_sharding_config(*, enable_ep: bool, enable_sp: bool) -> ShardingConfig:
    """
    ShardingConfig at the MoE boundary.

    Input boundary: S(1)@TP if SP, else I@TP.
    If SP+no EP, we all-gather -> R across sequence dimension.
    Otherwise the layout is preserved.
    """
    pre_moe_placement = (
        dense_sp_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I)
    )
    moe_placement = (
        dense_activation_placement(tp=spmd.R)
        if enable_sp and not enable_ep
        else pre_moe_placement
    )

    state_shardings = {
        "expert_bias_E": dense_param_placement(tp=spmd.R if enable_sp else spmd.I),
        "tokens_per_expert_E": _tokens_per_expert_placement(
            enable_ep=enable_ep, enable_sp=enable_sp
        ),
    }
    state_shardings_compute = {}
    if enable_ep and not enable_sp:
        state_shardings_compute["expert_bias_E"] = dense_param_placement(tp=spmd.R)

    return ShardingConfig(
        state_shardings=state_shardings,
        state_shardings_compute=state_shardings_compute,
        in_src_shardings={"x_BLD": pre_moe_placement},
        in_dst_shardings={"x_BLD": moe_placement},
        out_src_shardings=dense_activation_placement(tp=spmd.P),
        out_dst_shardings=pre_moe_placement,
    )


def set_moe_sharding_config(
    moe_cfg,
    *,
    enable_ep: bool,
    enable_sp: bool,
    expert_param_layout: dict[str, spmd.PerMeshAxisSpmdType],
    router_has_bias: bool = False,
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
    ``S(1)`` for colwise, ``S(2)`` for rowwise, ``I`` for
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

    # The MoE wrapper owns the EP-on/SP-off routed-path TP sequence split.
    moe_cfg.enable_ep = enable_ep
    moe_cfg.enable_sp = enable_sp
    moe_cfg.router.gate.sharding_config = _router_gate_config(
        enable_ep=enable_ep, enable_sp=enable_sp, has_bias=router_has_bias
    )

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
    experts_input_layout = (
        dense_sp_placement()
        if enable_ep
        else dense_activation_placement(tp=spmd.R if enable_sp else spmd.I)
    )
    experts_compute_layout = (
        dense_sp_placement()
        if enable_ep
        else dense_activation_placement(tp=spmd.R)
    )
    experts_out_layout = dense_activation_placement(tp=spmd.P)
    state_shardings_compute: dict[str, NamedPlacement] = {}
    if enable_ep:
        state_shardings: dict[str, NamedPlacement] = {
            name: expert_param_placement_sparse() for name in expert_param_layout
        }
    else:
        state_shardings = {
            name: expert_param_placement_dense(tp_placement=placement)
            for name, placement in expert_param_layout.items()
        }
        state_shardings_compute = {
            name: expert_param_placement_dense(tp_placement=spmd.R)
            for name, placement in expert_param_layout.items()
            if placement is spmd.I
        }

    moe_cfg.experts.sharding_config = ShardingConfig(
        state_shardings=state_shardings,
        state_shardings_compute=state_shardings_compute,
        in_src_shardings={
            "x_BLD": experts_input_layout,
            "topk_scores_BLK": experts_input_layout,
            "topk_expert_ids_BLK": experts_input_layout,
        },
        in_dst_shardings={
            "x_BLD": experts_compute_layout,
            "topk_scores_BLK": experts_compute_layout,
            "topk_expert_ids_BLK": experts_compute_layout,
        },
        out_src_shardings=experts_out_layout,
        local_spmd=True,
    )
