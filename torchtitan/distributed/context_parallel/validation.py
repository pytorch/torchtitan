# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copied from upstream open PR 4322/4449/4450 (fegin's CP stack) to unblock running; pending rebase and reconcile.

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

    cp = parallelism.context_parallel_degree
    tp = parallelism.tensor_parallel_degree

    if cp > 1 and parallelism.spmd_backend != "spmd_types":
        raise ValueError(
            "Context Parallel requires parallelism.spmd_backend='spmd_types', "
            f"got {parallelism.spmd_backend!r}."
        )

    # Decoder.prepare_batch uses the first full-attention kernel to decide
    # whether to shard the mask, then passes it to every full-attention layer.
    # All CP kernels must therefore use the same mask sharding.
    first_mask_sharding: tuple[str, bool] | None = None

    for fqn, traversed, _, _ in model.traverse(BaseAttention.Config):
        # traverse returns the base config type.
        attention = cast(BaseAttention.Config, traversed)
        # ``_owner`` is declared as a bare ``type``; it is the kernel class.
        kernel = cast("type[Module] | None", attention.inner_attention._owner)
        is_cp_kernel = kernel is not None and issubclass(kernel, ContextParallelKernel)

        if cp > 1 and not is_cp_kernel:
            raise ValueError(
                f"{fqn}.inner_attention must use a ContextParallelKernel, such "
                "as AllGatherCPFlexAttention, when the context parallel degree "
                "is larger than 1. Apply ContextParallelTransform; see an "
                "example in torchtitan_recipes/muse_glimmer.py."
            )
        if cp == 1 and is_cp_kernel:
            raise ValueError(
                f"{fqn}.inner_attention is a ContextParallelKernel but the "
                "context parallel degree is 1. Select a non-CP kernel."
            )

        shards_attention_heads = False
        if kernel is not None and is_cp_kernel:
            shards_attention_mask = getattr(kernel.Config, "shard_attention_mask", True)
            shards_attention_heads = getattr(
                kernel.Config, "shard_attention_heads", False
            )
            if first_mask_sharding is None:
                first_mask_sharding = (fqn, shards_attention_mask)
            elif first_mask_sharding[1] != shards_attention_mask:
                raise ValueError(
                    f"{fqn}.inner_attention and "
                    f"{first_mask_sharding[0]}.inner_attention "
                    "disagree on whether the attention mask is sharded, but one "
                    "mask is built for the whole model. Use CP kernels with the "
                    "same mask sharding."
                )

            if (
                not shards_attention_mask
                and parallelism.context_parallel_load_balancer is not None
            ):
                raise ValueError(
                    f"{fqn}.inner_attention uses {kernel.__qualname__}, which "
                    "keeps the attention mask global, so "
                    "context_parallel_load_balancer must be None. A load "
                    "balancer reorders the tokens but not the mask."
                )

        head_shard_degree = tp * cp if shards_attention_heads else tp
        divisor_description = (
            "tensor_parallel_degree * context_parallel_degree"
            if shards_attention_heads
            else "tensor_parallel_degree"
        )
        n_heads = attention.n_heads
        n_kv_heads = getattr(attention, "n_kv_heads", None) or n_heads
        for name, count in (("n_heads", n_heads), ("n_kv_heads", n_kv_heads)):
            if count % head_shard_degree != 0:
                raise ValueError(
                    f"{fqn}.inner_attention {name} ({count}) must be divisible "
                    f"by {divisor_description} ({head_shard_degree})."
                )
