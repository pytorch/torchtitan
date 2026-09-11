# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Validation across configuration components."""

from typing import cast, TYPE_CHECKING

from torchtitan.models.common.attention import BaseAttention

if TYPE_CHECKING:
    from torchtitan.config import ParallelismConfig
    from torchtitan.protocols.module import Module

__all__ = ["validate_context_parallel"]


def validate_context_parallel(
    model: "Module.Config", parallelism: "ParallelismConfig"
) -> None:
    """Validate that each inner attention matches the CP configuration."""
    from torchtitan.models.common.cp_attention import (
        CPInnerAttention,
        UlyssesCPFlexInnerAttention,
    )

    cp = parallelism.context_parallel_degree
    first_cp_attention: tuple[str, type[CPInnerAttention]] | None = None

    for fqn, traversed, _, _ in model.traverse(BaseAttention.Config):
        # traverse returns the base config type.
        attention = cast(BaseAttention.Config, traversed)
        owner = attention.inner_attention._owner
        is_cp_attention = owner is not None and issubclass(owner, CPInnerAttention)
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

        cp_attention = cast("type[CPInnerAttention]", owner)
        if first_cp_attention is None:
            first_cp_attention = (fqn, cp_attention)
        elif first_cp_attention[1] is not cp_attention:
            raise ValueError(
                f"{fqn}.inner_attention and "
                f"{first_cp_attention[0]}.inner_attention use different CP "
                "backends, but model inputs are sharded once."
            )
        # TODO(fegin): it seems to be cleaner if we move this logic to each
        # backend class definition. We need to revisit a good strategy to
        # define "where" should a validation implementation lives.
        if issubclass(cp_attention, UlyssesCPFlexInnerAttention):
            if parallelism.context_parallel_load_balancer is not None:
                raise ValueError(
                    f"{fqn}.inner_attention uses {cp_attention.__qualname__}, so "
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
