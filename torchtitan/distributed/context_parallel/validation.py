# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config-time validation for context parallelism."""

from typing import cast, TYPE_CHECKING

from torchtitan.models.common.attention import BaseAttention

if TYPE_CHECKING:
    from torchtitan.config import ParallelismConfig
    from torchtitan.protocols.module import Module

__all__ = ["validate_context_parallel"]


def validate_context_parallel(
    model: "Module.Config", parallelism: "ParallelismConfig"
) -> None:
    """Validate the CP backend and each attention kernel."""
    from torchtitan.models.common.cp_attention import ContextParallelKernel

    cp_enabled = parallelism.context_parallel_degree > 1
    if cp_enabled and parallelism.spmd_backend != "spmd_types":
        raise ValueError(
            "Context Parallel requires parallelism.spmd_backend='spmd_types', "
            f"got {parallelism.spmd_backend!r}."
        )

    for fqn, traversed, _, _ in model.traverse(BaseAttention.Config):
        # traverse returns the base config type.
        attention = cast(BaseAttention.Config, traversed)
        owner = attention.inner_attention._owner
        is_cp_kernel = owner is not None and issubclass(owner, ContextParallelKernel)
        if is_cp_kernel == cp_enabled:
            continue
        if cp_enabled:
            raise ValueError(
                f"{fqn}.inner_attention must use a ContextParallelKernel, such as "
                "AllGatherCPFlexAttention, when the context parallel degree is "
                "larger than 1. Apply ContextParallelTransform; see an example in "
                "torchtitan_recipes/muse_glimmer.py."
            )
        raise ValueError(
            f"{fqn}.inner_attention is a ContextParallelKernel but the context "
            "parallel degree is 1. Select a non-CP kernel."
        )
