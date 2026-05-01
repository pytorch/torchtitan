# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config-based sharding helpers for MoE submodules.

Mirrors the dense helpers in ``decoder_sharding.py`` for the MoE family.
Each MoE submodule's ``sharding_config`` mirrors the imperative legacy
plan (``apply_moe_ep_tp``) one-to-one:

- ``moe`` (wrapper): dense ``LocalMapConfig`` over ``{TP}``. Replaces
  ``PrepareModuleInputOutput(...)`` + ``MoE.forward``'s manual
  ``to_local(grad_placements=(Partial(),))`` at the dense<->local boundary.
- ``moe.router.gate``: ``Replicate`` weight, ``local_output_grad_placements
  ={TP: Partial()}``. Mirrors ``NoParallel(local_output_grad_placements=
  (Partial(),))``.
- ``moe.shared_experts.{w1,w3}``: colwise (``Shard(0)`` weight),
  ``local_input_grad_placements={"input": {TP: Partial()}}``,
  ``local_output_grad_placements`` set to make the wrapper return a local
  tensor (matching ``ColwiseParallelWithGradPlacement(use_local_output=
  True, local_input_grad_placements=(Partial(),))``).
- ``moe.shared_experts.w2``: rowwise (``Shard(1)`` weight),
  ``out_dst_shardings={TP: Partial()}`` and ``local_output_grad_placements
  ={TP: Partial()}``. Mirrors ``RowwiseParallel(output_layouts=Partial())``
  -- avoids the Partial->Replicate all-reduce in w2; the boundary collapse
  happens at the MoE wrapper.
- ``moe.experts`` (``GroupedExperts``): sparse ``{EP}`` placements
  when EP is enabled; falls back to dense ``{TP}`` placements when EP=1
  but TP>1 (today's ``TensorParallel`` path).

Family purity per Module is invariant: a Module's ``sharding_config``
references either dense ``{dp_replicate, dp_shard, cp, tp}`` axes or sparse
``{dp_replicate, efsdp, ep}`` axes, never both. The dense<->sparse
transition is the dispatcher's plain-tensor a2a, sitting *between* Modules
(``MoE`` exits dense, ``GroupedExperts`` enters sparse).
"""

from torch.distributed.tensor import Partial, Placement, Replicate, Shard

from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_param_placement,
)
from torchtitan.protocols.sharding import LocalMapConfig, NamedPlacement, ShardingConfig
from torchtitan.protocols.types import MeshAxisName


DP_REPLICATE = MeshAxisName.DP_REPLICATE
DP_SHARD = MeshAxisName.DP_SHARD
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
        DP_REPLICATE: Replicate(),
        EFSDP: Replicate(),
        EP: Shard(0),
    }


def expert_param_placement_dense(*, tp_placement: Placement) -> NamedPlacement:
    """Dense-family placement for routed-expert weights (EP=1, TP>1).

    Used when expert parallelism is disabled but tensor parallelism shards
    the experts on the dense TP axis (today's ``TensorParallel`` /
    ``GptossTensorParallel`` path).

    Insertion order matches canonical mesh order. ``tp_placement`` is the
    placement on the TP axis: ``Shard(1)`` for colwise, ``Shard(2)`` for
    rowwise, ``Replicate()`` for replicated bias.
    """
    return {
        DP_REPLICATE: Replicate(),
        DP_SHARD: Replicate(),
        CP: Replicate(),
        TP: tp_placement,
    }


def _shared_expert_colwise_config() -> ShardingConfig:
    """Colwise shared-expert FFN with Partial input gradient.

    Mirrors ``ColwiseParallelWithGradPlacement(use_local_output=True,
    local_input_grad_placements=(Partial(),))``: the input is wrapped as
    DTensor Replicate with grad_placements=Partial (so backward d_x stays
    Partial instead of all-reducing to Replicate); the output redistributes
    to ``Shard(1)`` (the hidden dim of the 2D ``(B*S, hidden)`` activation
    inside MoE) and unwraps to a local tensor for downstream.

    NOTE: ``Shard(1)`` is used explicitly instead of ``Shard(-1)`` because
    PyTorch's ``DTensor.to_local(grad_placements=...)`` requires normalized
    (non-negative) shard dims.
    """
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=Shard(0)),
            "bias": dense_param_placement(tp=Shard(0)),
        },
        in_src_shardings={"input": dense_activation_placement(tp=Replicate())},
        local_input_grad_placements={
            "input": dense_activation_placement(tp=Partial()),
        },
        out_dst_shardings=dense_activation_placement(tp=Shard(1)),
        local_output_grad_placements=dense_activation_placement(tp=Shard(1)),
    )


def _shared_expert_rowwise_config() -> ShardingConfig:
    """Rowwise shared-expert FFN with Partial output (no Partial->Replicate AR).

    Mirrors ``RowwiseParallel(output_layouts=Partial())`` + default
    ``use_local_output=True``: input is wrapped as DTensor Shard(-1) (the
    upstream colwise output's layout); the rowwise matmul produces Partial
    output; ``out_dst_shardings={TP: Partial()}`` keeps Partial (no
    redistribute, no AR); ``local_output_grad_placements`` unwraps to local
    with Partial backward annotation. The Partial->sp_layout reduce
    happens once at the MoE wrapper boundary.
    """
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=Shard(1)),
            # Rowwise bias is replicated (not sharded) -- the rowwise matmul
            # sums across TP ranks, then the bias add is a per-rank add.
            "bias": dense_param_placement(tp=Replicate()),
        },
        in_src_shardings={"input": dense_activation_placement(tp=Shard(1))},
        out_dst_shardings=dense_activation_placement(tp=Partial()),
        local_output_grad_placements=dense_activation_placement(tp=Partial()),
    )


def _router_gate_config() -> ShardingConfig:
    """Router gate: Replicate weight on TP, Partial output gradient on backward.

    Mirrors ``NoParallel(local_output_grad_placements=(Partial(),))``: the
    gate's weight is replicated on TP; in backward, the upstream local
    d_output is wrapped back as DTensor Partial on TP so the gate's
    matmul backward produces a Partial d_input that contributes to the
    accumulated MoE input gradient (collapsed at the wrapper boundary).
    """
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=Replicate()),
            "bias": dense_param_placement(tp=Replicate()),
        },
        in_src_shardings={"input": dense_activation_placement(tp=Replicate())},
        out_dst_shardings=dense_activation_placement(tp=Replicate()),
        local_output_grad_placements=dense_activation_placement(tp=Partial()),
    )


def _moe_wrapper_sharding_config(*, enable_sp: bool) -> ShardingConfig:
    """``ShardingConfig`` at the MoE wrapper boundary.

    Replaces ``PrepareModuleInputOutput(input_layouts=sp, desired_input_
    layouts=Replicate, output_layouts=Partial, desired_output_layouts=sp)``
    plus ``MoE.forward``'s manual ``to_local(grad_placements=(Partial(),))``.

    The flow under TP:
    - ``in_src_shardings`` declares the input arrives at ``sp_layout``.
    - ``in_dst_shardings`` redistributes input to ``Replicate`` (AG when
      SP is on; no-op otherwise).
    - ``LocalMapConfig`` then converts the Replicate DTensor input to a
      local tensor for the body. ``out_placements={TP: Partial()}`` wraps
      the body's local output (which is the per-rank Partial sum of all
      MoE contributions) as a ``Partial`` DTensor; ``in_grad_placements
      ={TP: Partial()}`` declares that the body's input gradient is also
      Partial on TP.
    - ``out_dst_shardings`` redistributes ``Partial -> sp_layout`` (RS
      when SP is on; AR when SP is off).

    Children (router, experts, shared_experts) re-wrap inputs to DTensors
    via their own ``sharding_config`` wrappers; the body operates on
    plain tensors at the call sites only.
    """
    sp_layout: Placement = Shard(1) if enable_sp else Replicate()
    return ShardingConfig(
        in_src_shardings={"x": dense_activation_placement(tp=sp_layout)},
        in_dst_shardings={"x": dense_activation_placement(tp=Replicate())},
        out_dst_shardings=dense_activation_placement(tp=sp_layout),
        local_map=LocalMapConfig(
            in_placements=(dense_activation_placement(tp=Replicate()),),
            out_placements=(dense_activation_placement(tp=Partial()),),
            in_grad_placements=(dense_activation_placement(tp=Partial()),),
        ),
    )


def set_moe_sharding_config(
    moe_cfg,
    *,
    tp_enabled: bool,
    ep_enabled: bool,
    enable_sp: bool,
    expert_param_layout: dict[str, Placement],
) -> None:
    """Populate ``sharding_config`` on every MoE submodule.

    Replaces the imperative ``apply_moe_ep_tp``. Branches dense vs sparse
    family per Module:

    - ``moe`` (wrapper): dense-family ``LocalMapConfig`` over ``{TP}``.
      Always set when ``tp_enabled`` -- even with EP, the wrapper's
      input/output activations are dense.
    - ``moe.router.gate``: dense-family TP plan with Partial output grad.
    - ``moe.shared_experts.{w1,w2,w3}``: dense-family TP plan with
      Partial-flow grad annotations (when ``moe_cfg.shared_experts is not
      None``).
    - ``moe.experts`` (``GroupedExperts``): sparse-family ``{EP}``
      placements when ``ep_enabled``; dense-family ``{TP}`` placements
      when ``ep_enabled is False`` and ``tp_enabled``.

    ``expert_param_layout`` maps each routed-expert parameter name to its
    dense in/out-dim placement (used when EP=1 + TP>1): ``Shard(1)`` for
    colwise, ``Shard(2)`` for rowwise, ``Replicate()`` for replicated
    bias. The shared ``GroupedExperts`` (qwen3, llama4, deepseek_v3) passes
    ``{"w1": Shard(1), "w2": Shard(2), "w3": Shard(1)}``;
    ``GptOssGroupedExperts`` passes its mlp1/mlp2 layout.

    Args:
        moe_cfg: The ``MoE.Config`` instance to populate.
        tp_enabled: Whether tensor parallelism is enabled.
        ep_enabled: Whether expert parallelism is enabled.
        enable_sp: Whether sequence parallelism is enabled (affects the
            wrapper's enter/exit TP layout).
        expert_param_layout: ``{param_name: tp_placement}`` for the
            routed experts' weight params (used in the EP=1 + TP>1 path).
    """
    # MoE wrapper + dense submodules. Set unconditionally (not gated on
    # ``tp_enabled``) so dense params land as DTensors on the dense SPMD
    # mesh under ``full_dtensor`` even when TP=1 -- otherwise FSDP2's
    # ``dp_mesh_dims`` invariant ("all params must be DTensors on the
    # full SPMD mesh") fires. Under TP=1, the TP-axis placements get
    # filtered out at ``resolve_placements`` time and the configs become
    # all-Replicate on the remaining dense axes (no-op for parallelism,
    # but the params are DTensors).
    moe_cfg.sharding_config = _moe_wrapper_sharding_config(enable_sp=enable_sp)

    # Router gate: dense-family TP plan with Partial output grad.
    moe_cfg.router.gate.sharding_config = _router_gate_config()

    # Shared experts: optional. Use Partial-flow variants so the
    # Partial->sp_layout reduce only happens once at the MoE boundary.
    if getattr(moe_cfg, "shared_experts", None) is not None:
        moe_cfg.shared_experts.w1.sharding_config = _shared_expert_colwise_config()
        moe_cfg.shared_experts.w2.sharding_config = _shared_expert_rowwise_config()
        moe_cfg.shared_experts.w3.sharding_config = _shared_expert_colwise_config()

    # Routed experts: branch on EP enablement to select the family.
    if ep_enabled:
        # Sparse family: {EP}. Expert weights shard on the expert dim.
        state_shardings: dict[str, NamedPlacement] = {
            name: expert_param_placement_sparse() for name in expert_param_layout
        }
        moe_cfg.experts.sharding_config = ShardingConfig(
            state_shardings=state_shardings,
        )
    elif tp_enabled:
        # Dense family: {TP}. EP=1 + TP>1 fallback (today's TensorParallel).
        state_shardings = {}
        for name, placement in expert_param_layout.items():
            state_shardings[name] = expert_param_placement_dense(
                tp_placement=placement,
            )
        moe_cfg.experts.sharding_config = ShardingConfig(
            state_shardings=state_shardings,
        )
    # else: EP=1 and TP=1 -- experts stay plain tensors, no sharding_config.
