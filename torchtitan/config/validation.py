# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Validation across configuration components."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torchtitan.models.common.attention import Attention, LinearAttention

if TYPE_CHECKING:
    from torchtitan.config import DebugConfig, TrainingConfig
    from torchtitan.config.parallelism import ParallelismConfig
    from torchtitan.distributed.activation_remat import ActivationRematConfig
    from torchtitan.distributed.parallelism_context import MeshAxisName
    from torchtitan.models.base.model import BaseModel
    from torchtitan.models.base.module import Module

__all__ = [
    "validate_context_parallel",
    "validate_model_training_config",
]


def _collect_model_layouts(
    model: Module.Config,
) -> list[tuple[str, str, set[MeshAxisName]]]:
    """Collect every SPMD layout declared by the model's sharding configs.

    Returns ``(fqn, field, axes)`` triples, where ``fqn`` mirrors the module
    FQN ``build()`` would produce and ``field`` names the ShardingConfig
    field holding the layout.
    """
    from torchtitan.distributed.spmd_types import spmd_axes
    from torchtitan.models.base.module import Module

    layouts = []
    # recurse=True: the root config itself is a Module.Config, and the
    # default stops descending into a matching subtree.
    for fqn, cfg, _, _ in model.traverse(Module.Config, recurse=True):
        sharding = cfg.sharding_config
        if sharding is None:
            continue
        fields = [
            (f"state_shardings[{key}]", layout)
            for key, layout in sharding.state_shardings.items()
        ]
        if sharding.in_src_shardings is not None:
            fields.extend(
                (f"in_src_shardings[{key}]", layout)
                for key, layout in sharding.in_src_shardings.items()
            )
        if sharding.in_dst_shardings is not None:
            fields.extend(
                (f"in_dst_shardings[{key}]", layout)
                for key, layout in sharding.in_dst_shardings.items()
            )
        out_src = sharding.out_src_shardings
        if out_src is not None:
            fields.extend(
                (f"out_src_shardings[{i}]", layout)
                for i, layout in enumerate(
                    out_src if isinstance(out_src, tuple) else (out_src,)
                )
            )
        if sharding.out_dst_shardings is not None:
            fields.append(("out_dst_shardings", sharding.out_dst_shardings))
        for field, layout in fields:
            layouts.append((fqn, field, set(spmd_axes(layout))))
    return layouts


def _layout_declares_axis(
    axis: MeshAxisName, axes: set[MeshAxisName], *, sparse: bool
) -> bool:
    """Whether a layout explicitly names an active parallelism axis."""
    from torchtitan.distributed.parallelism_context import MeshAxisName

    if axis in axes:
        return True
    if sparse:
        return False
    # Dense fwd/bwd layouts use logical ``dp``, which unfolds to storage axes.
    if axis in (MeshAxisName.DP_REPLICATE, MeshAxisName.DP_SHARD) and (
        MeshAxisName.DP in axes
    ):
        return True
    return False


def _dense_parallel_axes_required(
    parallelism: ParallelismConfig,
) -> tuple[MeshAxisName, ...]:
    """Mesh axes that must appear in every dense-region layout."""
    from torchtitan.distributed.parallelism_context import MeshAxisName

    required: list[MeshAxisName] = []
    if parallelism.tensor_parallel_degree > 1:
        required.append(MeshAxisName.TP)
    if parallelism.context_parallel_degree > 1:
        required.append(MeshAxisName.CP)
    if parallelism.data_parallel_replicate_degree > 1:
        required.append(MeshAxisName.DP_REPLICATE)
    dp_shard = parallelism.data_parallel_shard_degree
    if dp_shard > 1 or dp_shard == -1:
        required.append(MeshAxisName.DP_SHARD)
    return tuple(required)


def _sparse_parallel_axes_required(
    parallelism: ParallelismConfig,
) -> tuple[MeshAxisName, ...]:
    """Mesh axes that must appear in every sparse routed-expert layout."""
    from torchtitan.distributed.parallelism_context import MeshAxisName

    required: list[MeshAxisName] = [MeshAxisName.EP]
    if parallelism.data_parallel_replicate_degree > 1:
        required.append(MeshAxisName.DP_REPLICATE)
    dp_shard = parallelism.data_parallel_shard_degree
    if (
        dp_shard == -1
        or dp_shard
        * parallelism.context_parallel_degree
        * (parallelism.tensor_parallel_degree)
        // parallelism.expert_parallel_degree
        > 1
    ):
        required.append(MeshAxisName.EDP_SHARD)
    return tuple(required)


def _validate_declared_parallel_axes(
    model: Module.Config, parallelism: ParallelismConfig
) -> None:
    """Reject parallelism degrees the model cannot run.

    The check runs per layout, not over the union of all layouts: a layout
    that never mentions an axis is replicated over that axis, so a model
    that is only partly sharded must not pass.

    Dense and sparse layouts get different treatment.     Layouts on ``routed_experts`` submodules (``w13``, ``w2``, postprocess,
    and Dist-MoE children) live in the sparse ``{dp_replicate, edp_shard, ep}``
    region. Activation contracts on the ``routed_experts`` container and all
    other modules use the dense ``{dp, cp, tp}`` mesh. For each
    enabled degree in that region, every layout must name the axis explicitly.

    A config built with ``expert_parallel_degree == 1`` gets dense expert
    weight placements, so validating it at ``expert_parallel_degree > 1``
    fails here instead of silently ignoring the EP degree at runtime.
    """
    from torchtitan.distributed.parallelism_context import MeshAxisName
    from torchtitan.models.common.moe import MoE

    model_name = type(model).__qualname__
    ep = parallelism.expert_parallel_degree

    has_routed_experts = any(True for _ in model.traverse(MoE.Config))

    def _is_sparse(fqn: str) -> bool:
        """Layouts on routed-expert submodules use the sparse mesh.

        Activation contracts on the ``routed_experts`` container itself stay
        on the dense mesh.
        """
        parts = fqn.split(".")
        for index, part in enumerate(parts):
            if part != "routed_experts":
                continue
            return index + 1 < len(parts)
        return False

    dense_required = _dense_parallel_axes_required(parallelism)
    sparse_required = _sparse_parallel_axes_required(parallelism) if ep > 1 else ()

    if ep > 1 and not has_routed_experts:
        raise ValueError(
            f"{model_name} has no routed experts; set "
            "parallelism.expert_parallel_degree to 1 (got "
            f"{ep})."
        )

    for fqn, field, axes in _collect_model_layouts(model):
        axes_set = set(axes)
        if _is_sparse(fqn):
            required_axes = sparse_required
            sparse = True
        else:
            required_axes = dense_required
            sparse = False
        for axis in required_axes:
            if _layout_declares_axis(axis, axes_set, sparse=sparse):
                continue
            degree_field = {
                MeshAxisName.TP: "tensor_parallel_degree",
                MeshAxisName.CP: "context_parallel_degree",
                MeshAxisName.EP: "expert_parallel_degree",
                MeshAxisName.DP_REPLICATE: "data_parallel_replicate_degree",
                MeshAxisName.DP_SHARD: "data_parallel_shard_degree",
            }.get(axis)
            if degree_field is not None:
                degree = getattr(parallelism, degree_field)
                degree_msg = (
                    f"parallelism.{degree_field} is {degree}. Set {degree_field} to 1, "
                    "or declare the axis on that layout."
                )
            else:
                degree_msg = (
                    "parallelism enables a derived expert FSDP axis (edp_shard). "
                    "Declare 'edp_shard' on that layout."
                )
            raise ValueError(
                f"{model_name} declares no '{axis.value}' sharding in {fqn} ({field}), "
                f"but {degree_msg}"
            )


def validate_model_training_config(
    model: BaseModel.Config,
    *,
    parallelism: ParallelismConfig,
    training: TrainingConfig,
    debug: DebugConfig,
    activation_remat: ActivationRematConfig | None,
    max_num_documents: int | None,
) -> None:
    """Validate compatibility between a model and its training configuration."""
    from torchtitan.distributed.accelerator_graph import accelerator_graphs_supported
    from torchtitan.models.common.attention import VarlenInnerAttention
    from torchtitan.models.common.decoder import Decoder
    from torchtitan.models.common.moe import MoE
    from torchtitan.models.common.moe.token_dispatcher import (
        DeepEPTokenDispatcher,
        HybridEPTokenDispatcher,
        LocalTokenDispatcher,
    )

    _validate_declared_parallel_axes(model, parallelism)

    if isinstance(model, Decoder.Config):
        if model.enable_weight_tying and parallelism.pipeline_parallel_degree > 1:
            raise NotImplementedError(
                "Weight tying is not supported with Pipeline Parallel."
            )

        tp = parallelism.tensor_parallel_degree
        attention = model.first_full_attention
        if tp > 1 and attention is not None:
            num_heads = attention.n_heads
            num_kv_heads = getattr(attention, "n_kv_heads", None) or num_heads
            if num_heads % tp != 0:
                raise ValueError(
                    f"tensor_parallel_degree ({tp}) must divide n_heads ({num_heads})."
                )
            # Fused QKV projections shard whole KV-head groups. Attention with
            # separate K/V projections may instead shard each head's features.
            if hasattr(attention, "qkv_linear") and num_kv_heads % tp != 0:
                raise ValueError(
                    f"tensor_parallel_degree ({tp}) must divide "
                    f"n_kv_heads ({num_kv_heads})."
                )

        moe_configs = list(model.traverse(MoE.Config))
        ep = parallelism.expert_parallel_degree
        if moe_configs and ep < tp:
            raise ValueError(
                f"MoE models require expert_parallel_degree ({ep}) to be "
                f"greater than or equal to tensor_parallel_degree ({tp})."
            )
        for moe_fqn, moe, _, _ in moe_configs:
            if moe.num_experts % ep != 0:
                raise ValueError(
                    f"{moe_fqn}.num_experts ({moe.num_experts}) must be "
                    f"divisible by expert_parallel_degree ({ep})."
                )

        for fqn, dispatcher, _, _ in model.traverse(LocalTokenDispatcher.Config):
            if ep == 1 and isinstance(
                dispatcher,
                (DeepEPTokenDispatcher.Config, HybridEPTokenDispatcher.Config),
            ):
                raise ValueError(
                    f"{fqn} uses {type(dispatcher).__qualname__}, which requires "
                    "expert_parallel_degree greater than 1."
                )

        mtp_layers = getattr(model, "mtp_layers", None)
        if mtp_layers and parallelism.pipeline_parallel_degree > 1:
            raise NotImplementedError("MTP does not support pipeline parallelism yet.")

    if not training.disable_accelerator_graphs and accelerator_graphs_supported():
        if max_num_documents is None:
            for fqn, _, _, _ in model.traverse(VarlenInnerAttention.Config):
                raise ValueError(
                    "Accelerator graphs require fixed-shape varlen document "
                    f"metadata for {fqn}, but max_num_documents is unset. "
                    "Configure an upper bound on documents per local token "
                    "microbatch, or set training.disable_accelerator_graphs."
                )

        if parallelism.expert_parallel_degree > 1:
            for _, dispatcher_config, _, _ in model.traverse(
                LocalTokenDispatcher.Config
            ):
                if (
                    isinstance(dispatcher_config, HybridEPTokenDispatcher.Config)
                    and dispatcher_config.non_blocking_capacity_factor is not None
                ):
                    continue

                raise ValueError(
                    "Accelerator graphs support only expert parallel token dispatcher "
                    "configurations without CPU synchronization. "
                    "Set HybridEP non_blocking_capacity_factor, or set "
                    "training.disable_accelerator_graphs. Unsupported token "
                    f"dispatcher: {type(dispatcher_config).__qualname__}."
                )

    if debug.spmd_typechecking and model.local_compile.regions:
        # TODO: Remove this once Dynamo supports tracing SPMD typechecking.
        raise ValueError(
            "Local compilation is not supported with SPMD typechecking. "
            "Set model.local_compile.regions=[] or disable "
            "debug.spmd_typechecking."
        )

    validate_context_parallel(model, parallelism)


def validate_context_parallel(
    model: "Module.Config", parallelism: "ParallelismConfig"
) -> None:
    """Validate CP inner attentions, load balancers, and Ulysses head sharding."""
    from torchtitan.distributed.context_parallel import supports_cp_inner_attention
    from torchtitan.models.common.attention.cp_attention import (
        AllGatherKVVarlen,
        CPInnerAttention,
        Ulysses,
    )
    from torchtitan.models.common.attention.gdn import GDN
    from torchtitan.models.common.decoder import Decoder

    cp = parallelism.context_parallel_degree
    for fqn, traversed, _, _ in model.traverse(Attention.Config):
        attention = traversed
        inner_attention = attention.inner_attention
        is_cp_attention = isinstance(inner_attention, CPInnerAttention.Config)
        if cp > 1 and not is_cp_attention:
            raise ValueError(
                f"{fqn}.inner_attention must use CPInnerAttention, such as "
                "AllGatherKVFlexInnerAttention, when the context parallel degree is "
                "larger than 1. Apply ContextParallelTransform; see an example in "
                "torchtitan_recipes/models/muse_glimmer.py."
            )
        if cp == 1 and is_cp_attention:
            raise ValueError(
                f"{fqn}.inner_attention is CPInnerAttention but the "
                "context parallel degree is 1. Select a non-CP kernel."
            )
        if not is_cp_attention:
            continue

        cp_inner_attention = inner_attention._owner
        assert cp_inner_attention is not None and issubclass(
            cp_inner_attention, CPInnerAttention
        )
        if issubclass(cp_inner_attention, Ulysses):
            head_shard_degree = parallelism.tensor_parallel_degree * cp
            n_heads = attention.n_heads
            n_kv_heads = getattr(attention, "n_kv_heads", None) or n_heads
            if isinstance(attention, GDN.Config):
                n_kv_heads = attention.n_key_heads
            for name, count in (("n_heads", n_heads), ("n_kv_heads", n_kv_heads)):
                if count % head_shard_degree != 0:
                    raise ValueError(
                        f"{fqn}.inner_attention {name} ({count}) must be divisible "
                        "by tensor_parallel_degree * context_parallel_degree "
                        f"({head_shard_degree})."
                    )

        load_balancer_config = parallelism.context_parallel_load_balancer
        # Linear attention either routes states or restores token order using
        # the shared permutation; the full-attention backend selects the balancer.
        if isinstance(attention, LinearAttention.Config):
            continue
        if load_balancer_config is None:
            continue
        # Decoder ignores load balancers without full attention.
        # And Varlen rejects PTRR.
        if (
            isinstance(model, Decoder.Config)
            and model.first_full_attention is None
            and not issubclass(cp_inner_attention, AllGatherKVVarlen)
        ):
            continue
        if issubclass(cp_inner_attention, Ulysses):
            raise ValueError(
                f"{fqn}.inner_attention uses {cp_inner_attention.__qualname__}, so "
                "context_parallel_load_balancer must be None."
            )
        if not supports_cp_inner_attention(load_balancer_config, cp_inner_attention):
            raise ValueError(
                f"{type(load_balancer_config).__qualname__} does not support "
                f"{fqn}.inner_attention using {cp_inner_attention.__qualname__}."
            )
