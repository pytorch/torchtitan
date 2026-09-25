# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Validation across configuration components."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torchtitan.models.common.attention import BaseAttention

if TYPE_CHECKING:
    from torchtitan.config import (
        CompileConfig,
        DebugConfig,
        ParallelismConfig,
        TrainingConfig,
    )
    from torchtitan.distributed.activation_checkpoint import (
        ActivationCheckpointingConfig,
    )
    from torchtitan.protocols.module import Module

__all__ = ["validate_context_parallel", "validate_model_training_config"]


def validate_model_training_config(
    model: Module.Config,
    *,
    parallelism: ParallelismConfig,
    training: TrainingConfig,
    debug: DebugConfig,
    activation_checkpoint: ActivationCheckpointingConfig,
    compile_config: CompileConfig | None,
    max_num_documents: int | None,
) -> None:
    """Validate compatibility between a model and its training configuration."""
    from torchtitan.distributed.activation_checkpoint import MemoryBudgetAC
    from torchtitan.distributed.cuda_graph import cuda_graphs_supported
    from torchtitan.models.common.attention import VarlenInnerAttention
    from torchtitan.models.common.token_dispatcher import (
        HybridEPTokenDispatcher,
        LocalTokenDispatcher,
    )

    if not training.disable_cuda_graphs and cuda_graphs_supported():
        if max_num_documents is None:
            for fqn, _, _, _ in model.traverse(VarlenInnerAttention.Config):
                raise ValueError(
                    "CUDA graphs require fixed-shape varlen document "
                    f"metadata for {fqn}, but max_num_documents is unset. "
                    "Configure an upper bound on documents per local token "
                    "microbatch, or set --training.disable_cuda_graphs."
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
                    "CUDA graphs support only expert parallel token dispatcher "
                    "configurations without CPU synchronization. "
                    "Set HybridEP non_blocking_capacity_factor, or set "
                    "--training.disable_cuda_graphs. Unsupported token "
                    f"dispatcher: {type(dispatcher_config).__qualname__}."
                )

    if isinstance(activation_checkpoint, MemoryBudgetAC.Config) and not (
        compile_config is not None and "model" in compile_config.components
    ):
        raise ValueError(
            "Memory budget activation checkpointing requires the model to be "
            "compiled: configure CompileConfig and include 'model' in "
            "compile.components."
        )

    validate_context_parallel(model, parallelism)


def validate_context_parallel(
    model: "Module.Config", parallelism: "ParallelismConfig"
) -> None:
    """Validate that each inner attention matches the CP configuration."""
    from torchtitan.models.common.cp_attention import (
        CPInnerAttention,
        UlyssesCPInnerAttention,
    )

    cp = parallelism.context_parallel_degree
    first_cp_config: tuple[str, type] | None = None

    for fqn, traversed, _, _ in model.traverse(BaseAttention.Config):
        attention = traversed
        inner_attention = attention.inner_attention
        is_cp_attention = isinstance(inner_attention, CPInnerAttention.Config)
        if cp > 1 and not is_cp_attention:
            raise ValueError(
                f"{fqn}.inner_attention must use CPInnerAttention, such as "
                "KVAllGatherCPFlexInnerAttention, when the context parallel degree is "
                "larger than 1. Apply ContextParallelTransform; see an example in "
                "torchtitan_recipes/muse_glimmer.py."
            )
        if cp == 1 and is_cp_attention:
            raise ValueError(
                f"{fqn}.inner_attention is CPInnerAttention but the "
                "context parallel degree is 1. Select a non-CP kernel."
            )
        if not is_cp_attention:
            continue

        cp_config_type = type(inner_attention)
        if first_cp_config is None:
            first_cp_config = (fqn, cp_config_type)
        elif first_cp_config[1] is not cp_config_type:
            raise ValueError(
                f"{fqn}.inner_attention and "
                f"{first_cp_config[0]}.inner_attention use different CP "
                "backends, but model inputs are sharded once."
            )
        # TODO(fegin): it seems to be cleaner if we move this logic to each
        # backend class definition. We need to revisit a good strategy to
        # define "where" should a validation implementation lives.
        if isinstance(inner_attention, UlyssesCPInnerAttention.Config):
            if parallelism.context_parallel_load_balancer is not None:
                raise ValueError(
                    f"{fqn}.inner_attention uses {cp_config_type.__qualname__}, so "
                    "context_parallel_load_balancer must be None."
                )
            head_shard_degree = (
                parallelism.tensor_parallel_degree * parallelism.context_parallel_degree
            )
            n_heads = attention.n_heads
            n_kv_heads = getattr(attention, "n_kv_heads", None) or n_heads
            for name, count in (("n_heads", n_heads), ("n_kv_heads", n_kv_heads)):
                if count % head_shard_degree != 0:
                    raise ValueError(
                        f"{fqn}.inner_attention {name} ({count}) must be divisible "
                        "by tensor_parallel_degree * context_parallel_degree "
                        f"({head_shard_degree})."
                    )
