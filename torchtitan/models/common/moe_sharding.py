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


DP = MeshAxisName.DP
CP = MeshAxisName.CP
TP = MeshAxisName.TP
EP = MeshAxisName.EP
EFSDP = MeshAxisName.EFSDP
DP_REPLICATE = MeshAxisName.DP_REPLICATE


def expert_param_placement_sparse() -> NamedPlacement:
    """Sparse-family placement for routed-expert weights (EP enabled)."""
    return {
        DP_REPLICATE: spmd.R,
        EFSDP: spmd.R,
        EP: spmd.S(0),
    }


def expert_param_placement_dense(
    *, tp_placement: spmd.PerMeshAxisSpmdType
) -> NamedPlacement:
    """Dense-family placement for routed-expert weights (EP disabled, TP > 1)."""
    return dense_param_placement(tp=tp_placement)


def _shared_expert_colwise_config(enable_ep: bool, enable_sp: bool) -> ShardingConfig:
    """Colwise shared-expert FFN (w1/w3)."""
    del enable_ep, enable_sp
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.S(0)),
            "bias": dense_param_placement(tp=spmd.S(0)),
        },
        out_dst_shardings=dense_activation_placement(tp=spmd.S(-1)),
    )


def _shared_expert_rowwise_config() -> ShardingConfig:
    """Rowwise shared-expert FFN (w2), output stays Partial."""
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.S(1)),
            "bias": dense_param_placement(tp=spmd.I),
        },
        out_src_shardings=dense_activation_placement(tp=spmd.P),
        out_dst_shardings=dense_activation_placement(tp=spmd.P),
    )


def _shared_expert_state_config() -> ShardingConfig:
    """Shared expert state only, for runs without TP."""
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.I),
            "bias": dense_param_placement(tp=spmd.I),
        },
    )


def _router_gate_config(*, enable_ep: bool, has_bias: bool = False) -> ShardingConfig:
    """Router gate: replicated weights, output stays DTensor."""
    del enable_ep
    state_shardings = {"weight": dense_param_placement(tp=spmd.I)}
    state_shardings_compute = {"weight": dense_param_placement(tp=spmd.R)}
    if has_bias:
        state_shardings["bias"] = dense_param_placement(tp=spmd.I)
        state_shardings_compute["bias"] = dense_param_placement(tp=spmd.R)
    return ShardingConfig(
        state_shardings=state_shardings,
        state_shardings_compute=state_shardings_compute,
    )


def _tokens_per_expert_placement(*, enable_ep: bool) -> NamedPlacement:
    """Placement for the ``tokens_per_expert`` buffer."""
    del enable_ep
    return {
        DP: spmd.V,
        CP: spmd.V,
        TP: spmd.R,
    }


def _moe_sharding_config(*, enable_ep: bool, enable_sp: bool) -> ShardingConfig:
    """``ShardingConfig`` at the MoE boundary."""
    return ShardingConfig(
        state_shardings={
            "expert_bias": dense_param_placement(tp=spmd.R),
            "tokens_per_expert": _tokens_per_expert_placement(enable_ep=enable_ep),
        },
        in_src_shardings={
            "x": dense_sp_placement()
            if enable_sp
            else dense_activation_placement(tp=spmd.I)
        },
        in_dst_shardings={"x": dense_activation_placement(tp=spmd.R)},
        out_src_shardings=dense_activation_placement(tp=spmd.P),
        out_dst_shardings=(
            dense_sp_placement()
            if enable_sp
            else dense_activation_placement(tp=spmd.I)
        ),
        local_spmd=True,
    )


def set_moe_sharding_config(
    moe_cfg,
    *,
    enable_tp: bool,
    enable_cp: bool = False,
    enable_ep: bool,
    enable_sp: bool,
    expert_param_layout: dict[str, spmd.PerMeshAxisSpmdType] | None = None,
    router_has_bias: bool = False,
) -> None:
    """Populate ``sharding_config`` on every MoE submodule."""
    del enable_cp
    if expert_param_layout is None:
        expert_param_layout = {"w1": spmd.S(1), "w2": spmd.S(2), "w3": spmd.S(1)}

    # Always set sharding configs regardless of whether TP is enabled.
    # ``resolve_mesh`` filters out disabled axes at runtime.
    moe_cfg.sharding_config = _moe_sharding_config(
        enable_ep=enable_ep, enable_sp=enable_sp
    )

    moe_cfg.router.gate.sharding_config = _router_gate_config(
        enable_ep=enable_ep,
        has_bias=router_has_bias,
    )

    if getattr(moe_cfg, "shared_experts", None) is not None:
        if enable_tp:
            moe_cfg.shared_experts.w1.sharding_config = _shared_expert_colwise_config(
                enable_ep=enable_ep, enable_sp=enable_sp
            )
            moe_cfg.shared_experts.w2.sharding_config = _shared_expert_rowwise_config()
            moe_cfg.shared_experts.w3.sharding_config = _shared_expert_colwise_config(
                enable_ep=enable_ep, enable_sp=enable_sp
            )
        else:
            moe_cfg.shared_experts.w1.sharding_config = _shared_expert_state_config()
            moe_cfg.shared_experts.w2.sharding_config = _shared_expert_state_config()
            moe_cfg.shared_experts.w3.sharding_config = _shared_expert_state_config()

    if enable_ep:
        state_shardings: dict[str, NamedPlacement] = {
            name: expert_param_placement_sparse() for name in expert_param_layout
        }
        state_shardings_compute: dict[str, NamedPlacement] = {}
    elif enable_tp:
        state_shardings = {
            name: expert_param_placement_dense(tp_placement=placement)
            for name, placement in expert_param_layout.items()
        }
        state_shardings_compute = {
            name: expert_param_placement_dense(tp_placement=spmd.R)
            for name, placement in expert_param_layout.items()
            if name.endswith("bias") and placement is spmd.I
        }
    else:
        state_shardings = {
            name: expert_param_placement_dense(tp_placement=spmd.I)
            for name in expert_param_layout
        }
        state_shardings_compute = {}

    moe_cfg.experts.sharding_config = ShardingConfig(
        state_shardings=state_shardings,
        state_shardings_compute=state_shardings_compute,
    )
