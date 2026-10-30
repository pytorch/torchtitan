# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


@torch.library.custom_op("torchtitan::deterministic_scatter_add", mutates_args=())
def deterministic_scatter_add(
    out: torch.Tensor, index: torch.Tensor, src: torch.Tensor
) -> torch.Tensor:
    prev = torch.are_deterministic_algorithms_enabled()
    prev_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        return out.scatter_add(dim=0, index=index, src=src)
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn_only)


@deterministic_scatter_add.register_fake
def _(out: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(out)


def _backward(
    ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
) -> tuple[torch.Tensor, None, torch.Tensor]:
    (index,) = ctx.saved_tensors  # pyrefly: ignore[missing-attribute]
    grad_src = torch.gather(grad_output, dim=0, index=index)
    return grad_output, None, grad_src


def _setup_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    output: torch.Tensor,
) -> None:
    _out, index, _src = inputs
    ctx.save_for_backward(index)


deterministic_scatter_add.register_autograd(
    _backward,
    setup_context=_setup_context,
)


# Backends can override this op under their own dispatch key, e.g.
# "AutogradPrivateUse1", to run a fused combine kernel.
_lib = torch.library.Library("torchtitan", "FRAGMENT")
_lib.define(
    "score_and_scatter_add(Tensor routed, Tensor scores, Tensor token_indices, "
    "SymInt num_tokens) -> Tensor"
)


def _score_and_scatter_add_reference(
    routed: torch.Tensor,
    scores: torch.Tensor,
    token_indices: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    dim = routed.shape[-1]
    # Type promotion computes the product in float32 without materializing a
    # float32 copy of the routed output, which the multiply would save for
    # backward (twice the bytes of the routed output itself).
    scored = (routed * scores.reshape(-1, 1).to(torch.float32)).to(routed.dtype)
    return deterministic_scatter_add(
        routed.new_zeros(num_tokens, dim),
        token_indices.reshape(-1, 1).expand(-1, dim),
        scored,
    )


_lib.impl(
    "score_and_scatter_add",
    _score_and_scatter_add_reference,
    "CompositeImplicitAutograd",
)


def score_and_scatter_add(
    routed: torch.Tensor,
    scores: torch.Tensor,
    token_indices: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Weight routed rows by their router scores and sum them per token.

    Args:
        routed: ``(N, D)`` routed expert outputs.
        scores: ``(N,)`` router score of each routed row.
        token_indices: ``(N,)`` output row of each routed row.
        num_tokens: number of output rows.

    Returns:
        ``(num_tokens, D)`` combined output in ``routed.dtype``.
    """
    if torch.compiler.is_compiling():
        # Trace the plain ops so compiled graphs are unchanged.
        return _score_and_scatter_add_reference(
            routed, scores, token_indices, num_tokens
        )
    return torch.ops.torchtitan.score_and_scatter_add(
        routed, scores, token_indices, num_tokens
    )
