# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config-time validation for context parallelism.

Checks that need the actual attention mask stay at runtime, in ``cp_shard``
and in the kernels.
"""

from typing import cast, TYPE_CHECKING

from torchtitan.models.common.attention import BaseAttention

if TYPE_CHECKING:
    from torchtitan.config import ParallelismConfig
    from torchtitan.protocols.module import Module

__all__ = ["validate_context_parallel"]


def validate_context_parallel(
    model: "Module.Config", parallelism: "ParallelismConfig"
) -> None:
    """Check a model config against the parallelism config.

    Call this from a model config's ``update_from_config``. A model with no
    ``BaseAttention`` configs, such as Flux, gets the backend check only.
    """
    from torchtitan.models.common.cp_attention import ContextParallelKernel

    cp = parallelism.context_parallel_degree
    tp = parallelism.tensor_parallel_degree

    if cp > 1 and parallelism.spmd_backend != "spmd_types":
        raise ValueError(
            "Context Parallel requires parallelism.spmd_backend='spmd_types', "
            f"got '{parallelism.spmd_backend}'."
        )

    # One mask layout is built for the whole model, from the first
    # full-attention kernel, so every CP kernel must want the same one.
    mask_layout: tuple[str, bool] | None = None

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
                "is larger than 1. See an example in "
                "torchtitan_recipes/muse_glimmer.py."
            )
        if cp == 1 and is_cp_kernel:
            raise ValueError(
                f"{fqn}.inner_attention is a ContextParallelKernel but the "
                "context parallel degree is 1. Select a non-CP kernel."
            )

        # A kernel that keeps its mask global moved the CP shard off the tokens
        # and onto the heads. Both rules below follow from that: CP joins TP as
        # a head-count divisor, and a load balancer would reorder the tokens
        # while the global mask kept the original order.
        shards_heads = False
        if kernel is not None and is_cp_kernel:
            shards_mask = getattr(kernel.Config, "shard_attention_mask", True)
            if mask_layout is None:
                mask_layout = (fqn, shards_mask)
            elif mask_layout[1] != shards_mask:
                raise ValueError(
                    f"{fqn}.inner_attention and {mask_layout[0]}.inner_attention "
                    "disagree on whether the attention mask is sharded, but one "
                    "mask is built for the whole model. Use the same CP kernel "
                    "for every attention layer."
                )

            # A windowed layer is only caught at runtime otherwise, deep in
            # the kernel, after the model has already been built.
            if getattr(kernel.Config, "requires_causal_mask", False):
                window = getattr(attention.inner_attention, "window_size", (-1, 0))
                if window != (-1, 0):
                    raise ValueError(
                        f"{fqn}.inner_attention uses {kernel.__qualname__}, "
                        "which only supports causal masking under context "
                        f"parallel (window_size=(-1, 0)); got {window}."
                    )

            shards_heads = not shards_mask
            if shards_heads and parallelism.context_parallel_load_balancer is not None:
                raise ValueError(
                    f"{fqn}.inner_attention uses {kernel.__qualname__}, which "
                    "keeps the attention mask global, so "
                    "context_parallel_load_balancer must be None. A load "
                    "balancer reorders the tokens but not the mask."
                )

        divisor = tp * cp if shards_heads else tp
        axes = (
            "tensor_parallel_degree * context_parallel_degree"
            if shards_heads
            else "tensor_parallel_degree"
        )
        n_heads = attention.n_heads
        n_kv_heads = getattr(attention, "n_kv_heads", None) or n_heads
        for name, count in (("n_heads", n_heads), ("n_kv_heads", n_kv_heads)):
            if count % divisor != 0:
                raise ValueError(
                    f"{fqn}.inner_attention {name} ({count}) must be divisible "
                    f"by {axes} ({divisor})."
                )
