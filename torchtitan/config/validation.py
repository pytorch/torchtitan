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


def _declared_spmd_axes(model: Module.Config) -> set[MeshAxisName]:
    """Collect the mesh axes declared by the model's sharding configs.

    Each module declares the mesh axes it shards over in its
    ``sharding_config``; a model that never mentions an axis does not
    implement that form of parallelism.
    """
    from torchtitan.distributed.spmd_types import spmd_axes
    from torchtitan.models.base.module import Module

    declared = set()
    # recurse=True: the root config itself is a Module.Config, and the
    # default stops descending into a matching subtree.
    for _, cfg, _, _ in model.traverse(Module.Config, recurse=True):
        sharding = cfg.sharding_config
        if sharding is None:
            continue
        layouts = list(sharding.state_shardings.values())
        if sharding.in_src_shardings is not None:
            layouts.extend(sharding.in_src_shardings.values())
        if sharding.in_dst_shardings is not None:
            layouts.extend(sharding.in_dst_shardings.values())
        out_src = sharding.out_src_shardings
        if out_src is not None:
            layouts.extend(out_src if isinstance(out_src, tuple) else (out_src,))
        if sharding.out_dst_shardings is not None:
            layouts.append(sharding.out_dst_shardings)
        for layout in layouts:
            declared.update(spmd_axes(layout))
    return declared


def _validate_declared_parallel_axes(
    model: Module.Config, parallelism: ParallelismConfig
) -> None:
    """Reject parallelism degrees the model cannot run.

    Tensor parallelism is derived from the sharding configs: a model that
    shards no tensor over TP cannot run with ``tensor_parallel_degree > 1``
    (the degree would be silently ignored). TP placements are unconditional,
    so this does not depend on how the model config was built.

    Expert parallelism cannot be derived the same way: MoE weight placements
    only name the EP axis when the model config was built with EP enabled,
    so a config built with ``expert_parallel_degree == 1`` would look
    EP-incapable even for a genuine MoE model. Instead, EP support follows
    from the presence of MoE layers: a model with no MoE layers cannot run
    with ``expert_parallel_degree > 1``.
    """
    from torchtitan.distributed.parallelism_context import MeshAxisName
    from torchtitan.models.common.moe import MoE

    model_cls = type(model)._owner
    model_name = (
        model_cls.__name__ if model_cls is not None else type(model).__qualname__
    )
    if (
        parallelism.tensor_parallel_degree > 1
        and MeshAxisName.TP not in _declared_spmd_axes(model)
    ):
        raise ValueError(
            f"{model_name} declares no 'tp' sharding, but "
            "parallelism.tensor_parallel_degree is "
            f"{parallelism.tensor_parallel_degree}. Set tensor_parallel_degree "
            "to 1, or use a model that shards over 'tp'."
        )
    if parallelism.expert_parallel_degree > 1 and not list(model.traverse(MoE.Config)):
        raise ValueError(
            f"{model_name} has no MoE layers; set "
            f"parallelism.expert_parallel_degree to 1 (got "
            f"{parallelism.expert_parallel_degree})."
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
