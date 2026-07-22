# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config-based sharding helpers for MoE submodules."""

import spmd_types as spmd

from torchtitan.distributed.parallel_dims import MeshAxisName

from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
)
from torchtitan.models.common.token_dispatcher import TensorCombineResult
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig, SpmdLayout


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


def _shared_expert_colwise_config() -> ShardingConfig:
    """Colwise shared-expert FFN (w1/w3).

    Mirrors ``ColwiseParallel(input_layouts=...)``: input is all-gathered
    to Replicate for the column-sharded matmul; output is Shard(2)
    (feature dim for 3-D activations from MoE).
    """
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.S(0)),
            "bias": dense_param_placement(tp=spmd.S(0)),
        },
        in_src_shardings={"input": dense_activation_placement(tp=spmd.R)},
        in_dst_shardings={"input": dense_activation_placement(tp=spmd.R)},
        out_src_shardings=dense_activation_placement(tp=spmd.S(2)),
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
        out_src_shardings=dense_activation_placement(tp=spmd.P),
        out_dst_shardings=dense_activation_placement(tp=spmd.P),
    )


def _router_gate_config(*, enable_ep: bool, enable_sp: bool) -> ShardingConfig:
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
        input_layout = (
            dense_sequence_parallel_placement()
            if enable_sp
            else dense_activation_placement(tp=spmd.I)
        )
        output_layout = dense_sequence_parallel_placement()
        return ShardingConfig(
            state_shardings=state,
            in_src_shardings={"input": input_layout},
            in_dst_shardings={"input": output_layout},
            out_src_shardings=output_layout,
            out_dst_shardings=output_layout,
        )
    else:
        input_layout = dense_activation_placement(tp=spmd.R)
        return ShardingConfig(
            state_shardings=state,
            in_src_shardings={"input": input_layout},
            in_dst_shardings={"input": input_layout},
            out_src_shardings=input_layout,
            out_dst_shardings=input_layout,
        )


def _tokens_per_expert_placement(*, enable_ep: bool) -> SpmdLayout:
    """Placement for the ``tokens_per_expert_E`` buffer.

    Each DP/CP rank processes different data and accumulates partial token
    counts, so DP/CP axes are ``Partial``. TP is ``Partial`` when EP is
    enabled (MoE reuses the mesh axis named TP for sequence-token sharding, so
    each rank sees different tokens) or ``Replicate`` when EP is disabled (all
    TP ranks see the same tokens).
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
    the RoutedExperts boundary.
    """
    sp_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I)
    )
    moe_desired_input_layouts = (
        sp_layout if enable_ep else dense_activation_placement(tp=spmd.R)
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
    - ``moe.router.gate``: Replicate weights, output stays DTensor.
    - ``moe.shared_experts.{w1,w2,w3}``: dense-family TP plan with
      Partial-flow grad annotations (when ``moe_cfg.shared_experts is not
      None``).
    - ``moe.routed_experts.inner_experts`` (``GroupedExperts``): expert-weight
      ``state_shardings`` -- sparse ``{EP}`` / dense ``{TP}`` / none. The parent
      ``routed_experts`` holds the activation in/out shardings + local_map.

    ``expert_param_layout`` maps each routed-expert parameter name to its
    dense in/out-dim placement (used on the EP-disabled + TP-enabled path):
    ``Shard(1)`` for colwise, ``Shard(2)`` for rowwise, ``Replicate()`` for
    replicated bias. The shared ``GroupedExperts`` (qwen3, deepseek_v3)
    passes ``{"w1_EFD": Shard(1), "w2_EDF": Shard(2), "w3_EFD": Shard(1)}``;
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
    moe_cfg.seq_dim_tp_sharded = enable_ep and enable_sp
    moe_cfg.sharding_config = _moe_sharding_config(
        enable_ep=enable_ep, enable_sp=enable_sp
    )

    # Router gate: dense-family TP plan with Partial output grad.
    moe_cfg.router.gate.sharding_config = _router_gate_config(
        enable_ep=enable_ep, enable_sp=enable_sp
    )

    # Shared experts: SwiGLU FFN run in parallel with the routed experts.
    # Gather x to Replicate ONCE at the module boundary so w1/w3 share it (their
    # per-linear input redistributions become no-ops). w2 (rowwise) keeps the
    # output Partial; the Partial->sp_layout reduce happens once at the MoE
    # boundary. Model-specific shared-expert extras are sharded by that
    # model's own sharding code.
    shared = moe_cfg.shared_experts
    if shared is not None:
        # Shared-expert input matches the MoE input: sequence-parallel under
        # EP+SP (Shard(1) on seq, ordered via partition_spec), else Replicate.
        shared_input = (
            dense_sequence_parallel_placement()
            if enable_ep and enable_sp
            else dense_activation_placement(tp=spmd.I if enable_ep else spmd.R)
        )
        shared.sharding_config = ShardingConfig(
            in_src_shardings={"x": shared_input},
            in_dst_shardings={"x": dense_activation_placement(tp=spmd.R)},
        )

        shared.w1.sharding_config = _shared_expert_colwise_config()
        shared.w2.sharding_config = _shared_expert_rowwise_config()
        shared.w3.sharding_config = _shared_expert_colwise_config()

    # The three things that differ between EP and TP-only are the expert-weight
    # state_shardings, input layout, and input grad layout.
    experts_out_layout = dense_activation_placement(tp=spmd.P)
    if enable_ep:
        pre_experts_in_layout = (
            dense_sequence_parallel_placement()
            if enable_sp
            else dense_activation_placement(tp=spmd.I)
        )
        state_shardings: dict[str, SpmdLayout] = {
            name: expert_param_placement_sparse() for name in expert_param_layout
        }
        experts_in_layout = dense_sequence_parallel_placement()
        experts_in_grad_layout = dense_sequence_parallel_placement()
    else:
        pre_experts_in_layout = dense_activation_placement(tp=spmd.R)
        state_shardings = {
            name: expert_param_placement_dense(tp_placement=placement)
            for name, placement in expert_param_layout.items()
        }
        experts_in_layout = dense_activation_placement(tp=spmd.R)
        experts_in_grad_layout = dense_activation_placement(tp=spmd.P)

    # RoutedExperts (local_map region): activation in/out + local_map, no params.
    moe_cfg.routed_experts.sharding_config = ShardingConfig(
        in_src_shardings={
            "x_BLD": pre_experts_in_layout,
            "topk_scores_BLK": experts_in_layout,
            "topk_expert_ids_BLK": experts_in_layout,
            "num_local_tokens_per_expert_E": _tokens_per_expert_placement(
                enable_ep=enable_ep
            ),
        },
        in_dst_shardings={
            "x_BLD": experts_in_layout,
            "topk_scores_BLK": experts_in_layout,
            "topk_expert_ids_BLK": experts_in_layout,
            "num_local_tokens_per_expert_E": _tokens_per_expert_placement(
                enable_ep=enable_ep
            ),
        },
        out_src_shardings=TensorCombineResult(experts_out_layout),
        out_dst_shardings=None,
        local_map=LocalMapConfig(
            in_grad_placements=(
                (
                    experts_in_grad_layout,
                    experts_in_grad_layout,
                    experts_in_grad_layout,
                    # num_local_tokens_per_expert_E is routing metadata, but it is
                    # still a DTensor input to local_map and must have placements.
                    _tokens_per_expert_placement(enable_ep=enable_ep),
                )
                if enable_ep
                else None
            ),
        ),
    )

    # inner_experts owns the expert weights -> weight state_shardings only.
    moe_cfg.routed_experts.inner_experts.sharding_config = ShardingConfig(
        state_shardings=state_shardings,
    )
