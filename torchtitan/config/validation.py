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
        if sharding.input_shardings is not None:
            fields.extend(
                (f"input_shardings[{key}]", layout)
                for key, layout in sharding.input_shardings.items()
            )
        output = sharding.output_shardings
        if output is not None:
            fields.extend(
                (f"output_shardings[{i}]", layout)
                for i, layout in enumerate(
                    output if isinstance(output, tuple) else (output,)
                )
            )
        for field, layout in fields:
            layouts.append((fqn, field, set(spmd_axes(layout))))
    return layouts


def _validate_declared_parallel_axes(
    model: Module.Config, parallelism: ParallelismConfig
) -> None:
    """Reject parallelism degrees the model cannot run.

    The check runs per layout, not over the union of all layouts: a layout
    that never mentions an axis is replicated over that axis, so a model
    that is only partly sharded must not pass.

    Dense and sparse layouts get different treatment. When
    ``expert_parallel_degree > 1``, layouts on ``routed_experts``
    submodules (``w13``, ``w2``, postprocess, and Dist-MoE children)
    live in the sparse ``{dp_replicate, edp_shard, ep}`` region.
    When EP is off, those weights use the dense mesh, so they are
    checked as dense layouts. Activation contracts on the
    ``routed_experts`` container and all other modules always use the
    dense ``{dp, cp, tp}`` mesh. For each enabled degree in that
    region, every layout must name the axis explicitly.

    A config built with ``expert_parallel_degree == 1`` gets dense expert
    weight placements, so validating it at ``expert_parallel_degree > 1``
    fails here instead of silently ignoring the EP degree at runtime.
    """
    from torchtitan.distributed.parallelism_context import MeshAxisName
    from torchtitan.models.common.moe import MoE

    p = parallelism
    model_name = type(model).__qualname__
    # degree != 1 also covers dp_shard == -1.
    dense_required = [
        axis
        for axis, degree in (
            (MeshAxisName.TP, p.tensor_parallel_degree),
            (MeshAxisName.CP, p.context_parallel_degree),
            (MeshAxisName.DP_REPLICATE, p.data_parallel_replicate_degree),
            (MeshAxisName.DP_SHARD, p.data_parallel_shard_degree),
        )
        if degree != 1
    ]
    # The sparse region and the MoE check come from the same traversal, so
    # Dist-MoE (which swaps the routed_experts config type) is covered too.
    sparse_prefixes = tuple(
        f"{fqn}.routed_experts." if fqn else "routed_experts."
        for fqn, _, _, _ in model.traverse(MoE.Config)
    )
    sparse_required = []
    if p.expert_parallel_degree > 1:
        if not sparse_prefixes:
            raise ValueError(
                f"{model_name} has no routed experts; set "
                "parallelism.expert_parallel_degree to 1 (got "
                f"{p.expert_parallel_degree})."
            )
        # The mesh keeps edp_shard alive whenever EP is enabled.
        sparse_required = [MeshAxisName.EP, MeshAxisName.EDP_SHARD]
        if p.data_parallel_replicate_degree > 1:
            sparse_required.append(MeshAxisName.DP_REPLICATE)

    for fqn, field, axes in _collect_model_layouts(model):
        if p.expert_parallel_degree > 1 and fqn.startswith(sparse_prefixes):
            required = sparse_required
        else:
            required = dense_required
            if MeshAxisName.DP in axes:  # logical dp folds dp_replicate and dp_shard
                axes = axes | {MeshAxisName.DP_REPLICATE, MeshAxisName.DP_SHARD}
        for axis in required:
            if axis not in axes:
                raise ValueError(
                    f"{model_name} declares no '{axis.value}' sharding in "
                    f"{fqn} ({field}), but that mesh axis is active. Set its "
                    "parallelism degree to 1, or declare the axis on that layout."
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
    from torchtitan.distributed.context_parallel import (
        PTRRFlexAttentionCPLoadBalancer,
        supports_cp_inner_attention,
    )
    from torchtitan.models.common.attention.cp_attention import (
        AllGatherKVVarlen,
        CPInnerAttention,
        Ulysses,
        UlyssesInnerAttention,
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
            tp = parallelism.tensor_parallel_degree
            head_shard_degree = tp * cp
            n_heads = attention.n_heads
            n_kv_heads = getattr(attention, "n_kv_heads", None) or n_heads
            if isinstance(attention, GDN.Config):
                n_kv_heads = attention.n_key_heads
            if n_heads % head_shard_degree != 0:
                raise ValueError(
                    f"{fqn}.inner_attention n_heads ({n_heads}) must be divisible "
                    "by tensor_parallel_degree * context_parallel_degree "
                    f"({head_shard_degree})."
                )
            # UlyssesInnerAttention copies each TP-local KV head when there are
            # fewer KV heads than CP ranks.
            if n_kv_heads % head_shard_degree != 0 and not (
                issubclass(cp_inner_attention, UlyssesInnerAttention)
                and n_kv_heads % tp == 0
                and head_shard_degree % n_kv_heads == 0
            ):
                raise ValueError(
                    f"{fqn}.inner_attention n_kv_heads ({n_kv_heads}) must be "
                    "divisible by tensor_parallel_degree * context_parallel_degree "
                    f"({head_shard_degree}). UlyssesInnerAttention also accepts "
                    "n_kv_heads that divides tensor_parallel_degree * "
                    "context_parallel_degree and is divisible by "
                    "tensor_parallel_degree."
                )

        load_balancer_config = parallelism.context_parallel_load_balancer
        # Linear attention either routes states or restores token order using
        # the shared permutation; the full-attention backend selects the balancer.
        if isinstance(attention, LinearAttention.Config):
            # Linear attention costs the same per token, so PTRR's mask-based
            # balancing gains nothing, while its scattered token ranges inflate
            # the fixed CP routing capacity.
            if isinstance(load_balancer_config, PTRRFlexAttentionCPLoadBalancer.Config):
                raise ValueError(
                    f"{fqn}.inner_attention is linear attention, which does not "
                    "support PTRRFlexAttentionCPLoadBalancer; use "
                    "HeadTailCPLoadBalancer or no load balancer."
                )
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
