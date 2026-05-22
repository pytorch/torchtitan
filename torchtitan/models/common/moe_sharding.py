# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoE sharding configs for spmd_types-based model parallelism.

Defines sharding for the MoE wrapper boundary, router, shared experts,
and routed experts on both dense (TP) and sparse (EP) meshes.

Dense-path helpers (``dense_param_placement``, ``dense_activation_placement``)
are reused from ``decoder_sharding``. Sparse-path helpers for EP are defined
here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import spmd_types as spmd

from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_param_placement,
)
from torchtitan.protocols.sharding import (
    LocalSpmdConfig,
    NamedPlacement,
    ShardingConfig,
)
from torchtitan.protocols.types import MeshAxisName

if TYPE_CHECKING:
    from torchtitan.models.common.moe import MoE

DP = MeshAxisName.DP
CP = MeshAxisName.CP
TP = MeshAxisName.TP
EP = MeshAxisName.EP
EFSDP = MeshAxisName.EFSDP
DP_REPLICATE = MeshAxisName.DP_REPLICATE


# ---------------------------------------------------------------------------
# Sparse-mesh (EP) placement helpers
# ---------------------------------------------------------------------------


def sparse_param_placement() -> NamedPlacement:
    """Placement for expert params on the sparse mesh.

    DP_REPLICATE and EFSDP are ``Replicate`` at distribute time; FSDP
    reshards EFSDP post-parallelize. EP is ``S(0)`` — experts are always
    sharded on the expert dimension (dim 0).
    """
    return {
        DP_REPLICATE: spmd.R,
        EFSDP: spmd.R,
        EP: spmd.S(0),
    }


# ---------------------------------------------------------------------------
# MoE wrapper boundary
# ---------------------------------------------------------------------------


def moe_wrapper_config(*, enable_sp: bool) -> ShardingConfig:
    """Sharding at the MoE wrapper boundary (dense mesh).

    Input arrives at ``sp_layout`` on TP (S(1) when SP, I otherwise) and
    S(1) on CP. ``local_spmd`` runs the body on each local token shard, which
    keeps MoE's token flattening local under CP.
    """
    sp_tp = spmd.S(1) if enable_sp else spmd.I
    return ShardingConfig(
        state_shardings={
            "expert_bias": dense_param_placement(tp=spmd.R),
            "tokens_per_expert": dense_counter_placement(local=True),
        },
        in_src_shardings={"x": dense_activation_placement(tp=sp_tp)},
        in_dst_shardings={"x": dense_activation_placement(tp=spmd.R)},
        out_src_shardings=dense_activation_placement(tp=spmd.P),
        out_dst_shardings=dense_activation_placement(tp=sp_tp),
        local_spmd=LocalSpmdConfig(),
    )


def dense_counter_placement(*, local: bool) -> NamedPlacement:
    """Placement for counters accumulated from local token shards."""
    token_axis_type = spmd.V if local else spmd.P
    return {
        DP: token_axis_type,
        CP: token_axis_type,
        TP: spmd.R,
    }


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def router_gate_config(*, has_bias: bool = False) -> ShardingConfig:
    """Router gate: R@tp weight, R@tp activations.

    The gate's weight is replicated on TP. Computation is replicated.
    In backward, the upstream local grad is P@tp (from the MoE wrapper's
    local_spmd boundary), so the gate's weight grad is naturally P@tp
    and gets allreduced by FSDP post-backward (or convert(I,R) hook).
    """
    state_shardings = {"weight": dense_param_placement(tp=spmd.I)}
    state_tp_ir = {"weight"}
    if has_bias:
        state_shardings["bias"] = dense_param_placement(tp=spmd.I)
        state_tp_ir.add("bias")
    return ShardingConfig(
        state_shardings=state_shardings,
        state_tp_ir=state_tp_ir,
    )


# ---------------------------------------------------------------------------
# Shared experts (dense-mesh TP, Partial-flow)
# ---------------------------------------------------------------------------


def shared_expert_colwise_config() -> ShardingConfig:
    """Colwise shared-expert FFN: S(0)@tp weight, R@tp input, S(-1)@tp output.

    Input is R@tp (from MoE body). Colwise matmul shards the output
    hidden dim. In backward, input grad stays P@tp (no allreduce here —
    reduction happens once at the MoE wrapper boundary).
    """
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.S(0)),
            "bias": dense_param_placement(tp=spmd.S(0)),
        },
        out_dst_shardings=dense_activation_placement(tp=spmd.S(-1)),
    )


def shared_expert_state_config() -> ShardingConfig:
    """Shared expert state only, for runs without real dense TP."""
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.I),
            "bias": dense_param_placement(tp=spmd.I),
        },
    )


def shared_expert_rowwise_config() -> ShardingConfig:
    """Rowwise shared-expert FFN: S(1)@tp weight, S(1)@tp input, P@tp output.

    Input is S(-1)@tp (from colwise). Rowwise matmul produces P@tp.
    Output stays P@tp — no redistribution here. The MoE wrapper boundary
    handles the final P@tp → sp_layout reduction.
    """
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.S(1)),
            "bias": dense_param_placement(tp=spmd.I),
        },
        out_src_shardings=dense_activation_placement(tp=spmd.P),
        out_dst_shardings=dense_activation_placement(tp=spmd.P),
    )


# ---------------------------------------------------------------------------
# Routed experts
# ---------------------------------------------------------------------------


def routed_expert_ep_config(
    param_names: list[str],
    *,
    enable_tp: bool,
) -> ShardingConfig:
    """EP-enabled routed experts: S(0)@ep on sparse mesh.

    All expert params are sharded on the expert dimension (dim 0)
    across EP ranks. Dense/sparse activation transitions are explicit in
    the MoE dispatch/combine code.
    """
    return ShardingConfig(
        state_shardings={name: sparse_param_placement() for name in param_names},
    )


def routed_expert_tp_config(
    expert_param_layout: dict[str, spmd.PerMeshAxisSpmdType],
) -> ShardingConfig:
    """TP-only routed experts (EP disabled): TP-sharded on dense mesh.

    ``expert_param_layout`` maps param names to their TP shard placement:
    ``S(1)`` for colwise (w1, w3), ``S(2)`` for rowwise (w2).
    """
    return ShardingConfig(
        state_shardings={
            name: dense_param_placement(tp=tp_placement)
            for name, tp_placement in expert_param_layout.items()
        },
        state_tp_ir={
            name
            for name, tp_placement in expert_param_layout.items()
            if name.endswith("bias") and tp_placement is spmd.I
        },
    )


def routed_expert_state_config(param_names: list[str]) -> ShardingConfig:
    """Routed expert state only, for runs without TP or EP."""
    return ShardingConfig(
        state_shardings={
            name: dense_param_placement(tp=spmd.I) for name in param_names
        },
    )


# ---------------------------------------------------------------------------
# Top-level helper
# ---------------------------------------------------------------------------


def set_moe_sharding_config(
    moe_cfg: "MoE.Config",
    *,
    enable_tp: bool,
    enable_cp: bool = False,
    enable_ep: bool,
    enable_sp: bool,
    expert_param_layout: dict[str, spmd.PerMeshAxisSpmdType] | None = None,
    router_has_bias: bool = False,
) -> None:
    """Populate ``sharding_config`` on every MoE submodule.

    - ``moe`` (wrapper): dense-family ``LocalSpmdConfig`` for all configs.
    - ``moe.router.gate``: R@tp weight with I→R convert hook.
    - ``moe.shared_experts.{w1,w2,w3}``: dense-family TP with Partial flow.
    - ``moe.experts``: sparse-family EP (S(0)@ep) when EP enabled,
      dense-family TP when EP disabled.
    """
    if expert_param_layout is None:
        expert_param_layout = {"w1": spmd.S(1), "w2": spmd.S(2), "w3": spmd.S(1)}

    moe_cfg.sharding_config = moe_wrapper_config(enable_sp=enable_sp)
    moe_cfg.router.gate.sharding_config = router_gate_config(has_bias=router_has_bias)

    if getattr(moe_cfg, "shared_experts", None) is not None:
        if enable_tp:
            moe_cfg.shared_experts.w1.sharding_config = shared_expert_colwise_config()
            moe_cfg.shared_experts.w2.sharding_config = shared_expert_rowwise_config()
            moe_cfg.shared_experts.w3.sharding_config = shared_expert_colwise_config()
        else:
            moe_cfg.shared_experts.w1.sharding_config = shared_expert_state_config()
            moe_cfg.shared_experts.w2.sharding_config = shared_expert_state_config()
            moe_cfg.shared_experts.w3.sharding_config = shared_expert_state_config()

    if enable_ep:
        moe_cfg.experts.sharding_config = routed_expert_ep_config(
            list(expert_param_layout),
            enable_tp=enable_tp,
        )
    elif enable_tp:
        moe_cfg.experts.sharding_config = routed_expert_tp_config(expert_param_layout)
    else:
        moe_cfg.experts.sharding_config = routed_expert_state_config(
            list(expert_param_layout)
        )
